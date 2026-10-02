"""
Pipeline principal — RealMLP + SHAP + Regroupement hybride + Agent 1 +
Dense RAG + Agent 2, avec deux options de validation adversariale :
permutation du classement/scores/signes SHAP (``--permute-shap``), ou
injection d'une table SHAP CONSTANTE fictive (``--fake-shap-constant``,
même méthodologie que ``scripts/casinolimit/run_*_pipeline_adversarial.py``).

Utilisation en ligne de commande :

    python -m ids_pipeline.pipeline
    python -m ids_pipeline.pipeline --permute-shap
    python -m ids_pipeline.pipeline --fake-shap-constant --fake-shap-variant behavioral
    python -m ids_pipeline.pipeline --n-per-technique 10 --output results/run.json

Voir aussi ``scripts/05_run_shap_llm_pipeline.py`` (point d'entrée fin).
"""

import argparse
import json
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from ids_pipeline import config
from ids_pipeline.agents.agent1_behavior import run_agent1
from ids_pipeline.agents.agent2_ranking import run_agent2
from ids_pipeline.agents.rag_mitre import MitreDenseIndex, load_mitre_techniques
from ids_pipeline.explainability.shap_explainer import (
    build_correlation_matrix,
    compute_shap_values,
    create_explainer,
)
from ids_pipeline.explainability.fake_shap_zeek import FAKE_SHAP_TABLES_ZEEK
from ids_pipeline.explainability.shap_grouping import (
    aggregate_shap_by_correlation,
    build_explanation_text,
    build_feature_groups,
    permute_grouped_results,
)
from ids_pipeline.modeling.inference import (
    build_predict_fn,
    get_feature_columns,
    load_realmlp_model,
    prepare_realmlp_instance,
)

warnings.filterwarnings("ignore")


def select_test_events(
    df: pd.DataFrame,
    label_col: str,
    n_per_technique: int = config.NB_PAR_TECHNIQUE,
) -> pd.DataFrame:
    """Sélectionne ``n_per_technique`` événements pour chaque technique cible."""
    subsets = []
    for technique in config.TARGET_TECHNIQUES:
        subset = df[df[label_col].astype(str).str.contains(technique, case=False, na=False)]
        subsets.append(subset.head(n_per_technique))

    df_test = pd.concat(subsets, ignore_index=True)
    if len(df_test) == 0:
        raise ValueError("❌ Aucun événement cible trouvé (T1046/T1595).")
    return df_test


def find_label_column(df: pd.DataFrame) -> str:
    for col in config.POSSIBLE_LABEL_COLS:
        if col in df.columns:
            values = df[col].astype(str)
            if values.str.contains("T1046|T1595", case=False, regex=True, na=False).any():
                return col
    raise ValueError("❌ Impossible de trouver la colonne contenant T1046/T1595.")


def run_pipeline(
    dataset_path: str = config.FOCUS_DATASET_FILE,
    model_path: str = config.REALMLP_MODEL_FILE,
    mitre_json_path: str = config.MITRE_JSON_FILE,
    llm_model: str = config.LLM_MODEL,
    n_per_technique: int = config.NB_PAR_TECHNIQUE,
    permute_shap: bool = config.PERMUTE_SHAP_DEFAULT,
    permutation_seed: int = config.PERMUTATION_SEED_DEFAULT,
    fake_shap_constant: bool = False,
    fake_shap_variant: str = "behavioral",
) -> dict:
    rng = random.Random(permutation_seed)

    if fake_shap_constant and fake_shap_variant not in FAKE_SHAP_TABLES_ZEEK:
        raise ValueError(
            f"❌ fake_shap_variant doit être parmi {list(FAKE_SHAP_TABLES_ZEEK)}, "
            f"reçu : {fake_shap_variant!r}"
        )

    print("=" * 90)
    print("🚀 PIPELINE IDS — RealMLP + SHAP + HYBRID GROUPING + AGENTS LLM")
    if fake_shap_constant:
        print(f"🚨 MODE TEST : SHAP CONSTANT FICTIF (variante '{fake_shap_variant}') — "
              f"même explication injectée pour tous les événements, quel que soit leur contenu réel")
    elif permute_shap:
        print(f"⚠️ MODE TEST : SHAP PERMUTÉ (classement / scores / signes) — seed={permutation_seed}")
    else:
        print("✅ MODE NORMAL : SHAP réel (baseline)")
    print("=" * 90)

    # --- Chargement des données et du modèle ---
    print(f"\n📂 Chargement : {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    label_col = find_label_column(df)
    print(f"🏷️ Colonne utilisée pour les techniques : {label_col}")

    model = load_realmlp_model(model_path)
    feature_cols = get_feature_columns(model)
    predict_fn = build_predict_fn(model, feature_cols)

    # --- Sélection des événements de test ---
    df_test = select_test_events(df, label_col, n_per_technique)
    print(f"✅ TOTAL événements à analyser : {len(df_test)}")

    print("\n⏳ Préparation des événements pour RealMLP / SHAP...")
    X_test = pd.DataFrame(
        [prepare_realmlp_instance(row, feature_cols) for _, row in df_test.iterrows()]
    )[feature_cols]

    # --- Matrice de corrélation + groupes hybrides ---
    print("\n⏳ Calcul de la matrice de corrélation...")
    corr_matrix = build_correlation_matrix(df, feature_cols, lambda row: prepare_realmlp_instance(row, feature_cols))
    feature_groups = build_feature_groups(corr_matrix, feature_cols)

    print("\n📦 GROUPES HYBRIDES (SÉMANTIQUE + CORRÉLATION) TROUVÉS")
    for group_info in feature_groups:
        if len(group_info["features"]) > 1:
            print(f"[{group_info['semantic_name']}] " + " ↔ ".join(group_info["features"]))

    # --- Explainer SHAP ---
    print("\n⏳ Création de l'Explainer SHAP Permutation...")
    explainer = create_explainer(predict_fn, X_test.copy())
    print("✅ Explainer SHAP prêt !")

    # --- Base MITRE + index dense ---
    print("\n⏳ Chargement de la base MITRE et création des embeddings...")
    mitre_techniques = load_mitre_techniques(mitre_json_path)
    mitre_index = MitreDenseIndex(mitre_techniques)
    print(f"✅ Techniques MITRE indexées : {len(mitre_techniques)}")

    # --- Boucle principale ---
    score_agent2_top = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    score_dense_top15 = 0
    top15_frequency = Counter()
    results = []

    print("\n" + "=" * 90)
    print(f"🚀 ANALYSE DES {len(df_test)} ÉVÉNEMENTS")
    print("=" * 90)

    for index in range(len(df_test)):
        row = df_test.iloc[index]
        true_label = str(row[label_col])

        true_technique = next((t for t in config.TARGET_TECHNIQUES if t in true_label), None)
        if true_technique is None:
            continue

        print(f"\n🔍 Événement {index + 1}/{len(df_test)} | 🎯 Technique réelle : {true_technique}")

        X_instance = X_test.iloc[index:index + 1]

        probs = predict_fn(X_instance)[0]
        predicted_class_idx = int(np.argmax(probs))
        predicted_class = config.XGB_KNOWN_CLASSES[predicted_class_idx]
        predicted_probability = float(probs[predicted_class_idx])
        print(f"🤖 RealMLP : {predicted_class} ({predicted_probability * 100:.2f}%)")

        if fake_shap_constant:
            # Validation adversariale par table SHAP constante : la même
            # explication fictive est utilisée pour tous les événements,
            # sans aucun calcul SHAP réel sur cette instance.
            shap_text = FAKE_SHAP_TABLES_ZEEK[fake_shap_variant]
        else:
            try:
                shap_array = compute_shap_values(explainer, X_instance)
                grouped_results = aggregate_shap_by_correlation(shap_array, feature_cols, feature_groups)

                if permute_shap:
                    grouped_results = permute_grouped_results(grouped_results, rng)

                shap_text = build_explanation_text(grouped_results, max_groups=12)
            except Exception as e:
                print(f"❌ Erreur SHAP : {e}")
                continue

        try:
            behavior_description = run_agent1(shap_text, X_instance.iloc[0].to_dict(), llm_model)
        except Exception as e:
            print(f"❌ Erreur Agent 1 : {e}")
            continue

        print("\n📝 [AGENT 1]")
        print(behavior_description)

        dense_candidates = mitre_index.dense_retrieval(behavior_description, top_k=config.DENSE_RETRIEVAL_TOP_K)
        dense_top15_ids = [c["id"] for c in dense_candidates]

        if true_technique in dense_top15_ids:
            score_dense_top15 += 1
            top15_frequency[true_technique] += 1

        try:
            top5 = run_agent2(behavior_description, dense_candidates, llm_model)
        except Exception as e:
            print(f"❌ Erreur Agent 2 : {e}")
            top5 = []

        print("\n🏆 Agent 2 Top-5 :")
        for c in top5:
            print(f"   {c.get('rank')}. {c.get('technique_id')} — {c.get('technique_name')}")
            if c.get("technique_id") == true_technique and c.get("rank") in score_agent2_top:
                score_agent2_top[c.get("rank")] += 1

        results.append({
            "event_index": index + 1,
            "true_technique": true_technique,
            "realmlp_class": predicted_class,
            "behavior_description": behavior_description,
            "dense_top15": dense_top15_ids,
            "agent2_top5": top5,
        })

    # --- Rapport final ---
    total_events = len(results)
    print("\n" + "=" * 90)
    print("📊 RAPPORT FINAL")
    print("=" * 90)
    print(f"📌 Nombre d'événements analysés : {total_events}")
    if total_events:
        print(f"🔎 Dense RAG Top-K : {score_dense_top15}/{total_events} "
              f"({score_dense_top15 / total_events * 100:.2f}%)")
        print("🏆 Agent 2 :")
        for rank in range(1, 6):
            print(f"   Top-{rank} : {score_agent2_top[rank]}/{total_events} "
                  f"({score_agent2_top[rank] / total_events * 100:.2f}%)")

    return {
        "config": {
            "permute_shap": permute_shap,
            "permutation_seed": permutation_seed,
            "fake_shap_constant": fake_shap_constant,
            "fake_shap_variant": fake_shap_variant if fake_shap_constant else None,
            "n_per_technique": n_per_technique,
            "llm_model": llm_model,
        },
        "total_events": total_events,
        "score_dense_topk": score_dense_top15,
        "score_agent2_top": score_agent2_top,
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline IDS RealMLP + SHAP + Agents LLM")
    parser.add_argument("--dataset", default=config.FOCUS_DATASET_FILE)
    parser.add_argument("--model", default=config.REALMLP_MODEL_FILE)
    parser.add_argument("--mitre-json", default=config.MITRE_JSON_FILE)
    parser.add_argument("--llm-model", default=config.LLM_MODEL)
    parser.add_argument("--n-per-technique", type=int, default=config.NB_PAR_TECHNIQUE,
                         help="Nombre d'exemples testés par technique (pour limiter les appels "
                              "LLM), 5 par défaut. Remplacez cette valeur par le nombre souhaité.")
    parser.add_argument("--permute-shap", action="store_true", default=config.PERMUTE_SHAP_DEFAULT,
                         help="Active la validation adversariale (SHAP permuté)")
    parser.add_argument("--seed", type=int, default=config.PERMUTATION_SEED_DEFAULT)
    parser.add_argument("--fake-shap-constant", action="store_true", default=False,
                         help="Validation adversariale par table SHAP CONSTANTE (même explication "
                              "fictive injectée pour tous les événements, cf. scripts/casinolimit/"
                              "run_*_pipeline_adversarial.py). Mutuellement exclusif avec --permute-shap.")
    parser.add_argument("--fake-shap-variant", choices=["baseline", "behavioral"], default="behavioral",
                         help="Avec --fake-shap-constant : quelle table fictive utiliser, selon le "
                              "modèle (--model) chargé. Défaut : behavioral.")
    parser.add_argument("--output", default=None, help="Chemin du rapport JSON de sortie")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.fake_shap_constant and args.permute_shap:
        raise SystemExit("❌ --fake-shap-constant et --permute-shap sont mutuellement exclusifs.")

    report = run_pipeline(
        dataset_path=args.dataset,
        model_path=args.model,
        mitre_json_path=args.mitre_json,
        llm_model=args.llm_model,
        n_per_technique=args.n_per_technique,
        permute_shap=args.permute_shap,
        permutation_seed=args.seed,
        fake_shap_constant=args.fake_shap_constant,
        fake_shap_variant=args.fake_shap_variant,
    )

    output_path = args.output or f"{config.RESULTS_DIR}/run_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Rapport JSON sauvegardé dans : {output_path}")
    print("\n✅ PIPELINE TERMINÉ")


if __name__ == "__main__":
    main()
