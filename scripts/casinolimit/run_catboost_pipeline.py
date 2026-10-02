# ==============================================================================
# 🧠 PIPELINE COMPLET — CatBoost BASELINE et BEHAVIORAL, EXÉCUTÉS SÉPARÉMENT
# Chacun sur SON PROPRE échantillon de 10 (5 T1046 + 5 T1018, voir
# N_PER_TECHNIQUE ci-dessous)
# + SHAP CORRÉLATION-AWARE (tree_path_dependent) + Agent 1 + Dense RAG + Agent 2
#
# IMPORTANT :
# - Baseline et Behavioral sont deux pipelines INDÉPENDANTS, chacun avec :
#     * son propre échantillon de 10 événements (disjoints entre les deux)
#     * son propre modèle CatBoost
#     * son propre SHAP TreeExplainer en mode "tree_path_dependent", qui
#       exploite directement la structure de l'arbre et les fréquences de
#       passage observées à l'entraînement pour tenir compte nativement des
#       corrélations entre features (pas besoin de background, pas
#       d'hypothèse d'indépendance)
#     * son propre Agent 1, qui ne reçoit QUE les features de SON pipeline
#     * son propre Dense RAG + Agent 2 + scoring
# - La classe prédite / la confiance ne sont JAMAIS transmises aux LLM.
# 🆕 CORRELATION_THRESHOLD est désormais défini UNE SEULE FOIS en config
#    (0.60, ajusté suite à l'analyse de la matrice de corrélation réelle)
#    et réutilisé partout — plus de valeur codée en dur (0.80) éparpillée.
# 🆕 corr_matrix.fillna(0.0) gère les features à variance nulle (ex:
#    "duration" constant), qui produisaient des NaN et empêchaient toute
#    comparaison avec le seuil.
# ==============================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import sys
import logging
import warnings
import shap
import joblib
import catboost as cb

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ollama import chat


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

warnings.filterwarnings("ignore")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"

LLM_MODEL = "gpt-oss:120b-cloud"

# Chemins relatifs à la racine du dépôt (ce fichier vit dans scripts/casinolimit/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(REPO_ROOT, "data", "casinolimit")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "casinolimit")
MITRE_DIR = os.path.join(REPO_ROOT, "mitre")

INPUT_FILE = os.path.join(DATA_DIR, "final_dataset_14techniques_benign_behavioral.csv")

BASELINE_MODEL_FILE = os.path.join(MODEL_DIR, "catboost_binary_baseline.joblib")
BEHAVIORAL_MODEL_FILE = os.path.join(MODEL_DIR, "catboost_binary_behavioral.joblib")

MITRE_JSON_PATH = os.path.join(MITRE_DIR, "mitre_techniques_behavioral.json")

RANDOM_STATE = 42
N_PER_TECHNIQUE = 5  # Nombre d'exemples testés par technique (pour limiter les appels LLM).
                     # Pour changer le nombre d'essais, remplacez simplement ce 5 par le
                     # nombre souhaité.
TARGET_TECHNIQUES = ["T1046", "T1018", "T1125"]
CORRELATION_THRESHOLD = 0.60  # 🆕 seuil unique, ajusté suite à l'analyse de la matrice réelle

print("=" * 90)
print("🚀 PIPELINE CatBoost — BASELINE et BEHAVIORAL, séparés, SHAP corrélation-aware")
print("=" * 90)
print(f"✅ LLM configuré : {LLM_MODEL}")
print(f"✅ Seuil de corrélation utilisé : |r| >= {CORRELATION_THRESHOLD}")


# ==============================================================================
# 1. SUPPRESSION DES PRINTS
# ==============================================================================

class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ==============================================================================
# 2. FEATURES
# ==============================================================================

BASELINE_FEATURES = [
    "bytes",
    "packets",
    "duration",
    "proto_TCP",
    "proto_UDP",
    "proto_ICMP",
    "proto_GRE"
]

BEHAVIORAL_FEATURES = [
    "src_event_count_last_10",
    "src_unique_dst_last_10",
    "src_packets_sum_last_10",
    "src_bytes_sum_last_10",
    "src_duration_sum_last_10",
    "dst_unique_src_last_10",
    "new_destination",
    "repeated_connection",
    "time_since_previous_src_event",
    "time_since_previous_dst_event"
]

ALL_FEATURES = BASELINE_FEATURES + BEHAVIORAL_FEATURES

print(f"\n📐 Baseline features   : {len(BASELINE_FEATURES)} -> {BASELINE_FEATURES}")
print(f"📐 Behavioral features : {len(ALL_FEATURES)} -> {ALL_FEATURES}")


# ==============================================================================
# 3. CHARGEMENT DES MODELES CATBOOST
# ==============================================================================

print("\n⏳ Chargement des modèles CatBoost...")

# Chargement intelligent : tente d'abord joblib, puis la méthode native si erreur
try:
    cb_baseline = joblib.load(BASELINE_MODEL_FILE)
except Exception:
    cb_baseline = cb.CatBoostClassifier()
    cb_baseline.load_model(BASELINE_MODEL_FILE)

try:
    cb_behavioral = joblib.load(BEHAVIORAL_MODEL_FILE)
except Exception:
    cb_behavioral = cb.CatBoostClassifier()
    cb_behavioral.load_model(BEHAVIORAL_MODEL_FILE)

print(f"✅ Baseline   : {BASELINE_MODEL_FILE}")
print(f"✅ Behavioral : {BEHAVIORAL_MODEL_FILE}")

BINARY_CLASSES = ["BENIGN", "ATTACK"]


# ==============================================================================
# 3.bis VERIFICATION DES FEATURES ATTENDUES
# ==============================================================================

def get_cb_expected_features(model, fallback_cols):
    try:
        names = model.feature_names_
        if names:
            return list(names)
    except Exception:
        pass
    print("⚠️ Impossible de lire feature_names_ du modèle — fallback manuel utilisé.")
    return list(fallback_cols)


baseline_expected = get_cb_expected_features(cb_baseline, BASELINE_FEATURES)
behavioral_expected = get_cb_expected_features(cb_behavioral, ALL_FEATURES)

if set(baseline_expected) != set(BASELINE_FEATURES):
    print(f"⚠️ Attention : le modèle Baseline attend {baseline_expected}")
    BASELINE_FEATURES = baseline_expected

if set(behavioral_expected) != set(ALL_FEATURES):
    print(f"⚠️ Attention : le modèle Behavioral attend {behavioral_expected}")
    ALL_FEATURES = behavioral_expected

print(f"\n✅ Features Baseline confirmées   : {BASELINE_FEATURES}")
print(f"✅ Features Behavioral confirmées : {ALL_FEATURES}")


# ==============================================================================
# 4. CHARGEMENT DU DATASET
# ==============================================================================

print("\n⏳ Chargement du dataset complet...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"✅ {len(df):,} lignes chargées.")

possible_label_cols = ["label_technique", "technique_id", "label_id", "label"]
label_col = next((c for c in possible_label_cols if c in df.columns), None)

if label_col is None:
    raise ValueError(
        "❌ Aucune colonne de label de technique trouvée.\n"
        f"   Colonnes disponibles : {list(df.columns)}"
    )

print(f"✅ Colonne de technique utilisée : '{label_col}'")

for feature in ALL_FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df[feature] = df[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)

print("✅ Conversion numérique terminée.")

# 🆕 Diagnostic rapide : vérifier si des features ont une variance nulle
# (ce qui produit des NaN dans la matrice de corrélation)
zero_variance_features = [f for f in ALL_FEATURES if df[f].nunique() <= 1]
if zero_variance_features:
    print(f"\n⚠️ Features à variance nulle détectées (corrélation NaN pour ces colonnes) : "
          f"{zero_variance_features}")


# ==============================================================================
# 5. FONCTIONS DE PREDICTION
# ==============================================================================

def predict_fn_baseline(x):
    x = np.asarray(x, dtype=np.float64)
    df_x = pd.DataFrame(x, columns=BASELINE_FEATURES)
    with HiddenPrints():
        preds = cb_baseline.predict_proba(df_x)
    return np.asarray(preds)


def predict_fn_behavioral(x):
    x = np.asarray(x, dtype=np.float64)
    df_x = pd.DataFrame(x, columns=ALL_FEATURES)
    with HiddenPrints():
        preds = cb_behavioral.predict_proba(df_x)
    return np.asarray(preds)


# ==============================================================================
# 6. DESCRIPTIONS DES FEATURES
# ==============================================================================

FEATURE_DESCRIPTIONS = {

    "bytes":
        "the size of this specific network flow",
    "packets":
        "the number of packets in this specific flow",
    "duration":
        "the duration of this specific flow",

    "src_event_count_last_10":
        "the number of events this source has generated recently",
    "src_unique_dst_last_10":
        "the number of distinct destinations this source has recently contacted",
    "src_packets_sum_last_10":
        "the total packet volume sent by this source recently",
    "src_bytes_sum_last_10":
        "the total byte volume sent by this source recently",
    "src_duration_sum_last_10":
        "the total connection duration accumulated by this source recently",
    "dst_unique_src_last_10":
        "the number of distinct sources that have recently contacted this destination",
    "dst_event_count_last_10":
        "the number of events this destination has recently received",
    "new_destination":
        "whether this destination has not been contacted before by this source",
    "repeated_connection":
        "whether this connection repeats a previous one to the same destination",
    "time_since_previous_src_event":
        "the time elapsed since this source's previous event",
    "time_since_previous_dst_event":
        "the time elapsed since this destination's previous event",
}


# ==============================================================================
# 7. FONCTIONS GENERIQUES — SHAP TreeExplainer CORRÉLATION-AWARE
#    (tree_path_dependent) + REGROUPEMENT PAR CORRELATION
# ==============================================================================

def build_tree_explainer_for(cb_model, feature_list, df_source):
    """
    Construit un SHAP TreeExplainer propre à un modèle CatBoost et un jeu de
    features donné (Baseline OU Behavioral, jamais les deux mélangés).

    feature_perturbation="tree_path_dependent" exploite directement la
    structure de l'arbre et les statistiques de couverture observées à
    l'entraînement, ce qui tient compte nativement des corrélations entre
    features — sans supposer leur indépendance, et sans besoin de background.
    """
    df_numeric = df_source[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0)

    with HiddenPrints():
        explainer = shap.TreeExplainer(
            cb_model,
            feature_perturbation="tree_path_dependent"
        )

    return explainer, df_numeric


def build_feature_groups_for(feature_list, df_numeric, threshold=CORRELATION_THRESHOLD):
    """
    Regroupe par corrélation (composantes connexes) pour la présentation
    lisible côté Agent 1, UNIQUEMENT parmi les features de feature_list.

    🆕 threshold utilise désormais CORRELATION_THRESHOLD par défaut (0.60),
    au lieu de la valeur 0.80 codée en dur précédemment.
    🆕 .fillna(0.0) traite les colonnes à variance nulle (corrélation NaN,
    ex: "duration" constant) comme non corrélées, plutôt que de les laisser
    planter silencieusement la comparaison >= threshold.
    """
    corr_matrix = df_numeric[feature_list].corr().abs()
    corr_matrix = corr_matrix.fillna(0.0)

    feature_groups = []
    visited = set()

    for feature in feature_list:
        if feature in visited:
            continue

        group = set()
        stack = [feature]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)

            correlated_features = corr_matrix.loc[current][
                corr_matrix.loc[current] >= threshold
            ].index.tolist()

            for cf in correlated_features:
                if cf not in visited:
                    stack.append(cf)

        feature_groups.append(sorted(list(group)))

    return feature_groups


def aggregate_shap_by_correlation(shap_values_instance, feature_names, feature_groups):
    grouped_results = []

    for group_id, group in enumerate(feature_groups):
        indices = [feature_names.index(f) for f in group]
        values = np.asarray(shap_values_instance)[indices]

        total_signed = np.sum(values)
        total_abs = np.sum(np.abs(values))
        sign = "+" if total_signed > 0 else ("-" if total_signed < 0 else "0")

        grouped_results.append({
            "group_id": group_id,
            "features": group,
            "signed_shap": float(total_signed),
            "abs_shap": float(total_abs),
            "sign": sign
        })

    grouped_results.sort(key=lambda x: x["abs_shap"], reverse=True)
    return grouped_results


def build_explanation_text(grouped_results, feature_descriptions, max_groups=12):
    selected = grouped_results[:max_groups]
    explanation_lines = []

    for rank, item in enumerate(selected, start=1):
        group_features = item["features"]
        feature_descs = [feature_descriptions.get(f, f) for f in group_features]

        if len(group_features) == 1:
            feature_label = group_features[0]
        else:
            feature_label = "correlated feature group: " + ", ".join(group_features)

        description_text = " / ".join(feature_descs)

        explanation_lines.append(
            f"Rank {rank} | [{item['sign']}] | {feature_label} | "
            f"Behavioral meaning: {description_text}"
        )

    return "\n".join(explanation_lines)


# ==============================================================================
# 8. CHARGEMENT MITRE + EMBEDDINGS DENSES
# ==============================================================================

print(f"\n⏳ Chargement de la base MITRE ATT&CK ({MITRE_JSON_PATH})...")

with open(MITRE_JSON_PATH, "r", encoding="utf-8") as f:
    techniques = json.load(f)

print(f"✅ {len(techniques)} techniques MITRE chargées.")

print("\n⏳ Création de l'index Dense (all-MiniLM-L6-v2)...")

technique_texts = [
    t.get("behavioral_description", "") + " " + t.get("mitre_description", "")
    for t in techniques
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
technique_embeddings = embedding_model.encode(technique_texts, convert_to_tensor=False)

print(f"✅ {len(techniques)} techniques indexées.")


def retrieve_top15_dense(event_summary, techniques, embeddings_db, model):
    query_emb = model.encode([event_summary])[0]
    scores = cosine_similarity([query_emb], embeddings_db)[0]
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        tech = techniques[idx].copy()
        tech["dense_score"] = scores[idx]
        results.append(tech)
    return results


def prepare_candidates_for_agent2(top_k):
    candidates = []
    for item in top_k:
        desc = item.get("behavioral_description", "") + " " + item.get("mitre_description", "")
        candidates.append({
            "technique_id": item.get("technique_id", ""),
            "name": item.get("name", ""),
            "behavioral_description": desc
        })
    random.shuffle(candidates)
    return candidates


# ==============================================================================
# 9. ECHANTILLONNAGE — DEUX JEUX SÉPARÉS ET DISJOINTS DE 20 ÉVÉNEMENTS
# ==============================================================================

def sample_events(df_source, target_techniques, n_per_technique, label_col, random_state):
    sampled_dfs = []
    for tech in target_techniques:
        mask = df_source[label_col].astype(str).str.contains(tech, na=False)
        subset = df_source[mask]

        n_available = len(subset)
        n_take = min(n_per_technique, n_available)

        if n_available == 0:
            print(f"⚠️ Aucun exemple trouvé pour {tech}.")
            continue

        sampled = subset.sample(n=n_take, random_state=random_state)
        sampled_dfs.append(sampled)
        print(f"✅ {tech} : {n_take}/{n_available} exemples sélectionnés.")

    if not sampled_dfs:
        raise ValueError("❌ Aucun exemple trouvé pour aucune des techniques ciblées.")

    return pd.concat(sampled_dfs)


print("\n" + "=" * 90)
print("📊 ECHANTILLONNAGE BASELINE (20 événements)")
print("=" * 90)

baseline_events_df = sample_events(
    df, TARGET_TECHNIQUES, N_PER_TECHNIQUE, label_col, random_state=RANDOM_STATE
).reset_index(drop=False)

print(f"\n📦 Echantillon Baseline : {len(baseline_events_df)} événements.")

df_remaining = df.drop(index=baseline_events_df["index"].values)

print("\n" + "=" * 90)
print("📊 ECHANTILLONNAGE BEHAVIORAL (20 événements, disjoint de Baseline)")
print("=" * 90)

behavioral_events_df = sample_events(
    df_remaining, TARGET_TECHNIQUES, N_PER_TECHNIQUE, label_col, random_state=RANDOM_STATE
).reset_index(drop=False)

print(f"\n📦 Echantillon Behavioral : {len(behavioral_events_df)} événements.")

overlap = set(baseline_events_df["index"]) & set(behavioral_events_df["index"])
print(f"\n🔍 Chevauchement entre les deux échantillons : {len(overlap)} événement(s) "
      f"{'✅ (aucun, comme voulu)' if len(overlap) == 0 else '⚠️ ATTENTION'}")


# ==============================================================================
# 10. CONSTRUCTION DES SHAP TREEEXPLAINERS + GROUPES DE CORRELATION
#     🆕 threshold=CORRELATION_THRESHOLD au lieu de 0.80 codé en dur
# ==============================================================================

print("\n⏳ Construction du SHAP TreeExplainer (tree_path_dependent) — BASELINE...")

shap_explainer_baseline, df_numeric_baseline = build_tree_explainer_for(
    cb_baseline, BASELINE_FEATURES, df
)
feature_groups_baseline = build_feature_groups_for(
    BASELINE_FEATURES, df_numeric_baseline, threshold=CORRELATION_THRESHOLD
)

print(f"📦 Groupes de corrélation — BASELINE (seuil |r| >= {CORRELATION_THRESHOLD}) :")
for idx, group in enumerate(feature_groups_baseline, 1):
    print(f"   Groupe {idx} : " + (" ↔ ".join(group) if len(group) > 1 else group[0]))

print("\n⏳ Construction du SHAP TreeExplainer (tree_path_dependent) — BEHAVIORAL...")

shap_explainer_behavioral, df_numeric_behavioral = build_tree_explainer_for(
    cb_behavioral, ALL_FEATURES, df
)
feature_groups_behavioral = build_feature_groups_for(
    ALL_FEATURES, df_numeric_behavioral, threshold=CORRELATION_THRESHOLD
)

print(f"📦 Groupes de corrélation — BEHAVIORAL (seuil |r| >= {CORRELATION_THRESHOLD}) :")
for idx, group in enumerate(feature_groups_behavioral, 1):
    print(f"   Groupe {idx} : " + (" ↔ ".join(group) if len(group) > 1 else group[0]))


# ==============================================================================
# 11. FONCTION GENERIQUE — PIPELINE COMPLET SUR UN JEU DE FEATURES DONNE
# ==============================================================================

def run_full_pipeline(
    pipeline_label,
    events_df,
    predict_fn,
    feature_list,
    shap_explainer,
    feature_groups,
    feature_descriptions,
    target_techniques,
    llm_model,
    techniques_db,
    technique_embeddings_db,
    embedding_model_obj,
    label_col
):
    nb_events = len(events_df)
    score_detect = 0
    score_agent2_top = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    score_dense_top15 = 0
    top15_frequency = Counter()
    results = []

    print("\n")
    print("█" * 90)
    print(f"🚀 PIPELINE {pipeline_label} — {nb_events} événements — "
          f"{len(feature_list)} feature(s) : {feature_list}")
    print("█" * 90)

    for i, (_, row) in enumerate(events_df.iterrows(), start=1):

        real_technique = row[label_col]
        real_t_id = next(
            (t for t in target_techniques if t in str(real_technique)),
            str(real_technique)
        )

        print("\n" + "=" * 90)
        print(f"🚀 [{pipeline_label}] EVENEMENT {i}/{nb_events} | VRAIE TECHNIQUE : {real_technique}")
        print("=" * 90)

        # ------------------------------------------------------------------
        # A. INSTANCE (uniquement feature_list)
        # ------------------------------------------------------------------

        X_instance = (
            pd.to_numeric(row[feature_list], errors="coerce")
            .fillna(0).values.astype(np.float64).reshape(1, -1)
        )

        # ------------------------------------------------------------------
        # B. PREDICTION (affichage uniquement — jamais envoyée aux LLM)
        # ------------------------------------------------------------------

        probs = predict_fn(X_instance)[0]
        pred_class = BINARY_CLASSES[int(np.argmax(probs))]

        print(f"🔮 [{pipeline_label}] Prédiction : {pred_class:7s} (prob attack = {probs[1]:.4f})")

        if pred_class == "ATTACK":
            score_detect += 1

        # ------------------------------------------------------------------
        # C. SHAP TreeExplainer (tree_path_dependent — corrélation native)
        # ------------------------------------------------------------------

        print(f"\n🧠 [{pipeline_label}] Calcul SHAP (tree_path_dependent)...")

        try:
            df_instance = pd.DataFrame(X_instance, columns=feature_list)

            with HiddenPrints():
                shap_result = shap_explainer(df_instance)
                shap_values_all = shap_result.values

            if shap_values_all.ndim == 3:
                shap_values_instance = shap_values_all[0, :, 1]
            elif shap_values_all.ndim == 2:
                shap_values_instance = shap_values_all[0]
            else:
                shap_values_instance = np.asarray(shap_values_all).flatten()

        except Exception as e:
            print(f"❌ Erreur SHAP : {e}")
            continue

        # ------------------------------------------------------------------
        # D. REGROUPEMENT PAR CORRELATION
        # ------------------------------------------------------------------

        grouped_results = aggregate_shap_by_correlation(
            shap_values_instance, feature_list, feature_groups
        )

        shap_text = build_explanation_text(
            grouped_results, feature_descriptions, max_groups=12
        )

        print(f"\n🔎 [{pipeline_label}] SHAP (tree_path_dependent) + Correlation Grouping")
        print("-" * 70)
        print(shap_text)

        # ------------------------------------------------------------------
        # E. AGENT 1 — reçoit UNIQUEMENT feature_list
        # ------------------------------------------------------------------

        network_line = row[feature_list].to_dict()
        filtered_descriptions = {k: v for k, v in feature_descriptions.items() if k in feature_list}

        messages_agent1 = [
            {
                "role": "system",
                "content": f"""

You are a cybersecurity analyst specialized in Intrusion Detection Systems (IDS), network traffic analysis, and MITRE ATT&CK.

Your task is to transform a SHAP explanation of an IDS prediction into a concise, accurate, behavior-oriented description of the observed network activity.

The final description will be converted into a dense embedding and compared against MITRE ATT&CK technique descriptions. Therefore, your main objective is to preserve the most discriminative behavioral information contained in the network event and SHAP explanation without inventing intent or forcing the behavior into a specific MITRE technique.

IMPORTANT RULES:

* The output must be entirely in English.
* Do not reveal IP addresses.
* Do not reveal source or destination ports as numeric values. If port information is available, describe it only qualitatively, for example: "a well-known service port" or "a high-numbered ephemeral port".
* Do NOT name, predict, or suggest a MITRE ATT&CK technique or technique ID.
* Do NOT infer the ground-truth technique from the IDS prediction.
* Do NOT invent information that is not present in the network event or SHAP explanation.
* Do NOT assume that the activity is malicious.
* Do NOT automatically describe the activity as scanning, reconnaissance, discovery, credential use, command-and-control, collection, or exfiltration.
* Use such cybersecurity terminology ONLY when the observed network signals provide concrete behavioral evidence supporting it.
* Do not infer attacker intent when the available evidence is insufficient.
* Do not convert generic traffic characteristics into a specific attacker objective.
* Prefer observable behavior over interpretation of intent.

SHAP TREE EXPLANATION INTERPRETATION AND MANDATORY GROUNDING RULE:

SHAP (SHapley Additive exPlanations) explains the local IDS prediction by
computing each feature's exact contribution to the model's output for this
specific event, based on Shapley values derived from cooperative game
theory. For tree-based models, this is computed by tracing how each
feature's value shifts the prediction along the decision paths of the
model's trees, comparing against the expected output over the background
population. Unlike simple local approximations, SHAP values are additive:
the sum of all feature contributions plus a baseline expected value exactly
reconstructs the model's prediction for this event, so the ranked signals
below represent a mathematically consistent decomposition of the decision,
not an approximate or sampled estimate. The provided SHAP features are
ranked by the absolute magnitude of their contribution to the predicted IDS
class. A POSITIVE (+) contribution means the observed feature value pushed
the model's output toward the predicted class (ATTACK) relative to the
baseline expectation; a NEGATIVE (-) contribution means it pushed the
output away from the predicted class, which is still meaningful evidence
and must not be ignored, since it indicates the feature value is actually
less consistent with the predicted class than the average case. The SHAP
ranking describes what the model actually relied on for this decision, but
it does NOT indicate which feature is semantically most important for
cybersecurity or MITRE ATT&CK reasoning — a feature can rank highly simply
because its value is numerically unusual relative to the background
population, without being the most behaviorally meaningful signal — so do
not treat the highest-ranked feature as the main behavioral characteristic
automatically, and combine features that describe the same underlying
behavioral dimension (as indicated by correlated feature groups) rather
than treating them as independent facts. Every sentence you write in both
the EVENT SUMMARY and the BEHAVIOR DESCRIPTION must be traceable to a
specific ranked SHAP signal or correlated feature group listed above: before
writing, identify the top three to five ranked signals by absolute SHAP
magnitude, and make clear through your phrasing whether each characteristic
you describe reflects a positive or negative contribution — for example, a
positive SHAP value on destination diversity should be phrased as "the
source actively contacted N distinct destinations, a signal that pushed the
model toward its decision," whereas a negative SHAP value on that same
feature should be phrased as "despite contacting N destinations, this was
less unusual than typical attack traffic and pushed the model away from its
decision." Do not silently drop a top-ranked signal because it seems minor
or hard to phrase — if it has no natural cybersecurity interpretation, still
report the concrete observed value and its direction rather than omitting
it — and do not introduce any behavioral claim about frequency, diversity,
repetition, port type, or protocol pattern unless it corresponds to a signal
that actually appears in the ranked list; if a characteristic is not
supported by a ranked signal, leave it out entirely rather than inferring it
from the raw network event alone. A reader should be able to match every
claim in your description back to a specific rank and sign in the SHAP
explanation, and a description that could have been written from the raw
network event alone, without ever consulting the SHAP ranking, is not
acceptable.

Base your reasoning on the ranked SHAP signals, but the final text must describe pure observable behavior — no feature names, no "positive/negative contribution" language, no mention of SHAP, prediction, or model.

IMPORTANT DISTINCTION:

Separate MODEL EVIDENCE from BEHAVIORAL EVIDENCE.

MODEL EVIDENCE:
What network characteristics influenced the IDS prediction according to SHAP.

BEHAVIORAL EVIDENCE:
What those characteristics actually indicate about the observed communication pattern.

Your task is to translate the model evidence into behavioral evidence without adding unsupported conclusions.

FEATURE INTERPRETATION:

Use the provided feature descriptions to understand what each feature represents.

When several features describe the same behavioral dimension, combine them instead of repeating them independently.

For example:

Instead of:
"The event has 1 packet, 52 bytes, and zero duration."

Prefer:
"The event is a very small, short-lived network exchange consisting of a single packet."

TEMPORAL BEHAVIOR:

Pay particular attention to combinations of:

* time since previous source event
* time since previous destination event
* repeated connections
* event frequency
* recent packet counts
* recent byte counts

Describe whether the traffic appears:

* isolated or repeated
* sparse or frequent
* periodic or burst-like
* continuous or intermittent
* short-lived or long-lived

Do not call activity "automated" or "scripted" unless the combination of repeated, highly regular, or rapid events provides reasonable evidence for that interpretation.

DESTINATION / SOURCE BEHAVIOR:

Use source/destination diversity carefully.

For example:

* A high number of distinct destinations may indicate broader destination exploration.
* A low number of distinct destinations indicates focused communication.
* A high number of distinct sources communicating with one destination indicates a different communication pattern.
* A new destination indicates a change in the source's communication pattern.
* A repeated destination does NOT by itself indicate scanning.

Do not infer scanning or discovery from a single connection or from small packet sizes alone.

PROTOCOL:

Describe the protocol when it provides useful behavioral context.

Do NOT assume:

TCP = scanning
UDP = discovery
ICMP = reconnaissance
GRE = malicious activity

The protocol must be interpreted together with the other behavioral signals.

VOCABULARY FOR DOWNSTREAM SEMANTIC RETRIEVAL:

The description will later be compared semantically with MITRE ATT&CK descriptions.

Use precise cybersecurity vocabulary when supported by the evidence, such as:

* repeated communication
* short-lived connection
* low-volume traffic
* high-frequency communication
* destination diversity
* source diversity
* internal communication
* service interaction
* host interaction
* probing
* enumeration
* discovery
* authentication activity
* credential-related activity
* periodic communication
* data transfer

However, DO NOT force any of these concepts into the description.

Choose only the behavioral vocabulary supported by the combination of features.

IMPORTANT:
Do not write a generic sentence such as:
"The activity appears to be reconnaissance."

Instead, describe the concrete behavioral evidence that could support or contradict such an interpretation.

For example:
"The source repeatedly contacted several distinct internal destinations within a short time window using small, short-lived connections."

This preserves the behavioral evidence without prematurely assigning an attacker objective.

CRITICAL ANTI-HALLUCINATION RULE:

If the available network signals do not provide enough evidence to distinguish between multiple possible behaviors, remain neutral.

It is better to say:

"The traffic consists of repeated short-lived communications with a limited set of internal destinations."

than to say:

"The attacker is scanning the internal network."

unless the network evidence clearly supports the latter interpretation.

The goal is not to guess the attacker's intention.

The goal is to accurately represent the observable network behavior in a form that preserves useful semantic information for downstream retrieval.

SHAP EXPLANATION:

{shap_text}

OUTPUT FORMAT:

Your response MUST contain exactly two sections.

1. EVENT SUMMARY

Write ONE concise paragraph describing the individual network event and its immediate context.

Do not state an attacker objective.

2. BEHAVIOR DESCRIPTION

Write ONE SINGLE PARAGRAPH describing the broader observable network behavior.

Do NOT use bullet points inside either section.

Do NOT include analysis of the SHAP method itself in the final output.

Do NOT mention these instructions.

Return only:

EVENT SUMMARY

[one paragraph]

BEHAVIOR DESCRIPTION

[one paragraph]
"""
            },
            {
                "role": "user",
                "content": f"""
Analyze the following network event, but strictly follow this CRITICAL OVERRIDE DIRECTIVE:
The SHAP EXPLANATION reveals the true, underlying malicious behavior detected by the advanced AI. The raw NETWORK EVENT data provided is obfuscated, spoofed, or incomplete. 
If there is ANY contradiction between the physical NETWORK EVENT values and the SHAP EXPLANATION, you MUST prioritize the SHAP EXPLANATION.


NETWORK EVENT:
{network_line}

SHAP EXPLANATION:
{shap_text}

FEATURE DESCRIPTIONS:
{filtered_descriptions}
"""}]

        try:
            resp1 = chat(model=llm_model, messages=messages_agent1)
            summary = resp1.get("message", {}).get("content", "") if isinstance(resp1, dict) else resp1.message.content
        except Exception as e:
            print(f"❌ Erreur Agent 1 : {e}")
            summary = ""

        print(f"\n📝 [{pipeline_label}] AGENT 1 — BEHAVIOR DESCRIPTION")
        print("-" * 70)
        print(summary)

        # ------------------------------------------------------------------
        # F. DENSE RAG TOP-15
        # ------------------------------------------------------------------

        all_candidates = retrieve_top15_dense(
            summary, techniques_db, technique_embeddings_db, embedding_model_obj
        )
        top15_candidats = all_candidates[:25]
        top15_ids = [str(c.get("technique_id", "")) for c in top15_candidats]

        print(f"\n🔎 [{pipeline_label}] DENSE RAG TOP-15")
        for rank, c in enumerate(top15_candidats, 1):
            print(f"{rank:2d}. {c.get('technique_id','')} — {c.get('name','')}")

        for t_id in top15_ids:
            top15_frequency[t_id] += 1

        if any(real_t_id in t_id for t_id in top15_ids):
            print(f"\n✅ [{pipeline_label}] DENSE RETRIEVAL : {real_t_id} dans le Top-15")
            score_dense_top15 += 1
        else:
            print(f"\n❌ [{pipeline_label}] DENSE RETRIEVAL : {real_t_id} absent du Top-15")
            results.append({
                "event_index": i, "true_technique": real_t_id,
                "pred_class": pred_class, "behavior_description": summary,
                "dense_top15": top15_ids, "agent2_top5": []
            })
            continue

        # ------------------------------------------------------------------
        # G. AGENT 2
        # ------------------------------------------------------------------

        agent2_candidates = prepare_candidates_for_agent2(top15_candidats)

        candidates_text_dynamic = ""
        for idx_c, candidate in enumerate(agent2_candidates, start=1):
            desc = candidate.get("behavioral_description", "")
            candidates_text_dynamic += (
                f"Candidate {idx_c}\nTechnique ID: {candidate['technique_id']}\n"
                f"Name: {candidate['name']}\nDescription: {desc}\n\n"
            )

        agent2_prompt = f"""
You are a cybersecurity analyst specializing in
MITRE ATT&CK and network traffic analysis.

Determine which 5 candidate techniques are best supported
by the observed behavior below.

Rank the five techniques from the closest to the farthest.

========================
OBSERVED BEHAVIOR
========================
{summary}

========================
CANDIDATE TECHNIQUES
========================
{candidates_text_dynamic}

CRITICAL INSTRUCTIONS:
1. Base your ranking STRICTLY on the candidate descriptions provided above.
2. DO NOT rely on external knowledge beyond the candidate descriptions.
3. If a technique is not in the candidate list, you cannot select it.
4. Do not assume that the first candidate is correct.
5. The candidate order is random.
6. Return exactly five candidates.

Return ONLY valid JSON:
{{
    "top_5": [
        {{"rank": 1, "technique_id": "...", "technique_name": "...", "reason": "..."}},
        {{"rank": 2, "technique_id": "...", "technique_name": "...", "reason": "..."}},
        {{"rank": 3, "technique_id": "...", "technique_name": "...", "reason": "..."}},
        {{"rank": 4, "technique_id": "...", "technique_name": "...", "reason": "..."}},
        {{"rank": 5, "technique_id": "...", "technique_name": "...", "reason": "..."}}
    ]
}}
"""

        try:
            resp2 = chat(model=llm_model, messages=[{"role": "user", "content": agent2_prompt}])
            prediction_text = resp2.get("message", {}).get("content", "") if isinstance(resp2, dict) else resp2.message.content
        except Exception as e:
            print(f"❌ Erreur Agent 2 : {e}")
            prediction_text = ""

        if "```json" in prediction_text:
            prediction_text = prediction_text.split("```json")[1].split("```")[0].strip()
        elif "```" in prediction_text:
            prediction_text = prediction_text.split("```")[1].split("```")[0].strip()

        top_5 = []

        try:
            prediction_json = json.loads(prediction_text)
            top_5 = prediction_json.get("top_5", [])
            top_5_ids = [str(t.get("technique_id", "")) for t in top_5]

            print(f"\n🤖 [{pipeline_label}] AGENT 2 — Top 5 :")
            for rank, t_id in enumerate(top_5_ids, start=1):
                print(f"   {rank}. {t_id}")

            rang_trouve = -1
            for r, t_id in enumerate(top_5_ids, start=1):
                if real_t_id == t_id:
                    rang_trouve = r
                    break

            if rang_trouve != -1:
                print(f"\n🎯 [{pipeline_label}] SUCCÈS Agent 2 : {real_t_id} trouvé au rang {rang_trouve}")
                for k in range(rang_trouve, 6):
                    score_agent2_top[k] += 1
            else:
                print(f"\n❌ [{pipeline_label}] ÉCHEC Agent 2 : {real_t_id} absent du Top-5")

        except json.JSONDecodeError as e:
            print(f"❌ Erreur parsing JSON Agent 2 : {e}")

        results.append({
            "event_index": i, "true_technique": real_t_id,
            "pred_class": pred_class, "behavior_description": summary,
            "dense_top15": top15_ids, "agent2_top5": top_5
        })

    return {
        "pipeline_label": pipeline_label,
        "nb_events": nb_events,
        "score_detect": score_detect,
        "score_dense_top15": score_dense_top15,
        "score_agent2_top": score_agent2_top,
        "top15_frequency": top15_frequency,
        "results": results
    }


# ==============================================================================
# 12. EXECUTION — DEUX PIPELINES INDEPENDANTS
# ==============================================================================

baseline_output = run_full_pipeline(
    pipeline_label="BASELINE",
    events_df=baseline_events_df,
    predict_fn=predict_fn_baseline,
    feature_list=BASELINE_FEATURES,
    shap_explainer=shap_explainer_baseline,
    feature_groups=feature_groups_baseline,
    feature_descriptions=FEATURE_DESCRIPTIONS,
    target_techniques=TARGET_TECHNIQUES,
    llm_model=LLM_MODEL,
    techniques_db=techniques,
    technique_embeddings_db=technique_embeddings,
    embedding_model_obj=embedding_model,
    label_col=label_col
)

behavioral_output = run_full_pipeline(
    pipeline_label="BEHAVIORAL",
    events_df=behavioral_events_df,
    predict_fn=predict_fn_behavioral,
    feature_list=ALL_FEATURES,
    shap_explainer=shap_explainer_behavioral,
    feature_groups=feature_groups_behavioral,
    feature_descriptions=FEATURE_DESCRIPTIONS,
    target_techniques=TARGET_TECHNIQUES,
    llm_model=LLM_MODEL,
    techniques_db=techniques,
    technique_embeddings_db=technique_embeddings,
    embedding_model_obj=embedding_model,
    label_col=label_col
)


# ==============================================================================
# 13. RAPPORT FINAL — LES DEUX PIPELINES, SEPAREMENT
# ==============================================================================

def print_report(output):
    label = output["pipeline_label"]
    nb_events = output["nb_events"]
    results = output["results"]

    print("\n" + "█" * 90)
    print(f"🚀 RAPPORT DE PERFORMANCE — PIPELINE {label}")
    print("█" * 90)

    print(f"\n📊 1. DETECTION BINAIRE (ATTACK vs BENIGN)")
    print(f"   ► {output['score_detect']}/{nb_events} "
          f"({output['score_detect']/nb_events*100:.1f}%)")

    print(f"\n🔍 2. DENSE RETRIEVAL")
    print(f"   ► Top-15 : {output['score_dense_top15']}/{nb_events} "
          f"({output['score_dense_top15']/nb_events*100:.1f}%)")

    print(f"\n🤖 3. PIPELINE COMPLET (Agent 2)")
    for k in range(1, 6):
        acc_k = output["score_agent2_top"][k] / nb_events * 100
        print(f"   ► Top-{k} : {output['score_agent2_top'][k]}/{nb_events} ({acc_k:.1f}%)")

    print(f"\n📋 4. DETAIL PAR TECHNIQUE")
    for tech in TARGET_TECHNIQUES:
        tech_results = [r for r in results if r["true_technique"] == tech]
        n_tech = len(tech_results)
        if n_tech == 0:
            continue

        detect_ok = sum(1 for r in tech_results if r["pred_class"] == "ATTACK")
        dense_ok = sum(1 for r in tech_results if tech in r["dense_top15"])
        top1_ok = sum(
            1 for r in tech_results
            if r["agent2_top5"] and r["agent2_top5"][0].get("technique_id") == tech
        )

        print(f"\n{tech} ({n_tech} exemples) :")
        print(f"   Détection      : {detect_ok}/{n_tech}")
        print(f"   Dense RAG Top-15 : {dense_ok}/{n_tech}")
        print(f"   Agent 2 Top-1  : {top1_ok}/{n_tech}")

    print(f"\n🌍 5. DIVERSITÉ DU RETRIEVAL DENSE")
    freq = output["top15_frequency"]
    print(f"   ► Techniques différentes remontées : {len(freq)} sur {len(techniques)} possibles")
    print("\n   ► Techniques les plus fréquemment remontées :")
    for t_id, count in freq.most_common(10):
        print(f"      - {t_id} : {count} apparitions")


print_report(baseline_output)
print_report(behavioral_output)

print("\n" + "█" * 90)
print("✅ LES DEUX PIPELINES CatBoost (BASELINE et BEHAVIORAL) SONT TERMINÉS")
print("█" * 90)
