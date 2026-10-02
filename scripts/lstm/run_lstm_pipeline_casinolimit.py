#!/usr/bin/env python3
"""
Pipeline complet CasinoLimit : LSTM (déjà entraîné) -> xNIDS (Sparse
Group Lasso, auto-contenu) -> Agent 1 (LLM, synthèse comportementale)
-> RAG dense MITRE ATT&CK -> Agent 2 (LLM, classement Top-5).

Contrairement au pipeline SHAP+Agents (``scripts/casinolimit/run_*_pipeline.py``),
l'explicabilité xNIDS est calculée DYNAMIQUEMENT pour chaque séquence
(aucun fichier précalculé requis) — voir ``scripts/lstm/README.md``.

Pré-requis : avoir lancé ``scripts/lstm/train_lstm_casinolimit.py``
(modèle non fourni dans cette archive, voir ``models/casinolimit/README.md``).

Usage:
    python scripts/lstm/run_lstm_pipeline_casinolimit.py --n-attacks 5
"""

import argparse
import json
import os
import re
import sys
import warnings

import joblib
import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casinolimit_data import build_sequences, load_dataframe, transform_padded
from lstm_common import LSTMBinary, XNIDSExplainer, build_top_features_text

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "casinolimit")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "casinolimit")
MITRE_DIR = os.path.join(REPO_ROOT, "mitre")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

SRC = os.path.join(DATA_DIR, "final_dataset_14techniques_benign_behavioral.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_timeseries.pt")
SCALER_FILE = os.path.join(MODEL_DIR, "lstm_timeseries_scaler.joblib")
MITRE_JSON = os.path.join(MITRE_DIR, "mitre_techniques_behavioral.json")

LLM_MODEL = os.environ.get("IDS_LLM_MODEL", "gpt-oss:120b-cloud")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_DISPLAY_COLS = ["timestamp", "protocol", "bytes", "packets", "duration", "src_ip", "dst_ip"]

AGENT1_SYSTEM_PROMPT = """
You are a cybersecurity analyst specialized in Intrusion Detection Systems (IDS) and MITRE ATT&CK.

Analyze the provided sequence of network flows and its feature importance explanation matrix.
- The event is a sequence of up to 10 time steps, from t-N to t, where 't' is the most recent malicious event.
- The explanation matrix shows the feature importance score at each time step. Positive/high values strongly contributed to the attack detection by the ML model.

Produce an English description of the observed behavior.
Describe what the attacker is doing based on the traffic sequence and the important features.
Do NOT use bullet points.
Do NOT name, predict, or suggest a MITRE ATT&CK technique or technique ID
Return only:

EVENT SUMMARY
[one paragraph]

BEHAVIOR DESCRIPTION
[one paragraph]
"""


def load_model():
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)  # checkpoint de confiance (le nôtre), contient des listes/dicts Python en plus des tensors
    model = LSTMBinary(
        n_features=checkpoint["n_features"],
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.2),
        bidirectional=checkpoint.get("bidirectional", True),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def select_attack_sequences(y, teams, n_per_dataset, seed=42):
    # Tirage ALEATOIRE (et non plus "la premiere occurrence par equipe") :
    # l'ordre de construction de X place les fenetres a historique court
    # (n_real proche de MIN_SEQ_LEN) en tete de chaque equipe, donc prendre
    # systematiquement "la premiere attaque trouvee" biaisait l'echantillon
    # vers le cas le plus rare (0.03% des sequences d'attaque reelles ont
    # n_real=3, contre 99.85% a n_real=WINDOW=10).
    attack_indices = np.where(y == 1)[0]
    if len(attack_indices) == 0:
        raise RuntimeError("Aucune attaque dans les séquences disponibles.")

    rng = np.random.RandomState(seed)
    shuffled_indices = attack_indices.copy()
    rng.shuffle(shuffled_indices)

    selected, used_teams = [], set()
    for idx in shuffled_indices:
        team_key = str(teams[idx])
        if team_key not in used_teams:
            selected.append(int(idx))
            used_teams.add(team_key)
        if len(selected) == n_per_dataset:
            break
    return selected


def format_flows(df, row_indices):
    available = [c for c in RAW_DISPLAY_COLS if c in df.columns]
    flows = df.loc[row_indices, available].reset_index(drop=True)
    n = len(flows)
    flows.index = [f"t-{n - 1 - j}" if j < n - 1 else "t" for j in range(n)]
    return flows


def run_agent1(flows_text, explanation_text):
    from ollama import chat

    messages = [
        {"role": "system", "content": AGENT1_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"NETWORK FLOWS (Sequence of events):\n{flows_text}\n\n"
            f"EXPLANATION (xNIDS top features, higher is more important):\n{explanation_text}"
        )},
    ]
    resp = chat(model=LLM_MODEL, messages=messages)
    content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else resp.message.content
    return content.strip()


def load_mitre_index():
    from sentence_transformers import SentenceTransformer

    with open(MITRE_JSON, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)

    techniques = []
    for tech in mitre_data:
        full_desc = (tech.get("behavioral_description", "") + " " + tech.get("mitre_description", "")).strip()
        techniques.append({"id": tech.get("technique_id", ""), "name": tech.get("name", ""), "description": full_desc})

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{t['id']} {t['name']} {t['description']}" for t in techniques]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return techniques, model, embeddings


def dense_retrieval(behavior_description, techniques, model, embeddings, top_k=25):
    query_emb = model.encode([behavior_description], normalize_embeddings=True, show_progress_bar=False)[0]
    similarities = np.dot(embeddings, query_emb)
    ranking = np.argsort(similarities)[::-1][:top_k]
    return [techniques[i] for i in ranking]


def run_agent2(behavior_description, candidates):
    from ollama import chat

    candidates_text = "".join(
        f"Candidate {i}\nID: {c['id']}\nName: {c['name']}\nDescription: {c['description']}\n\n"
        for i, c in enumerate(candidates, 1)
    )
    prompt = f"""
You are a cybersecurity analyst. Select and rank the top 5 most likely candidates from the provided list using ONLY their descriptions.
Return exactly five candidates in JSON format.

Behavior description:
{behavior_description}

Candidates list:
{candidates_text}

Return ONLY valid JSON:
{{ "top_5": [ {{"rank": 1, "technique_id": "TXXXX", "technique_name": "...", "reason": "..."}} ] }}
"""
    resp = chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else resp.message.content
    match = re.search(r"\{.*\}", content, re.DOTALL)
    return json.loads(match.group(0)).get("top_5", []) if match else []


def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-attacks", type=int, default=5,
                         help="Nombre de séquences attaque analysées (une par équipe), pour "
                              "limiter les appels LLM. Remplacez ce 5 par le nombre souhaité.")
    parser.add_argument("--top-n-features", type=int, default=12, help="Nombre de features xNIDS retenues pour Agent 1")
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "lstm_casinolimit_run.json"))
    args = parser.parse_args()

    LLM_MODEL = args.llm_model

    print("=" * 90)
    print("🚀 Pipeline LSTM CasinoLimit — xNIDS + Agent 1 + RAG dense + Agent 2")
    print("=" * 90)

    if not os.path.exists(MODEL_FILE):
        raise SystemExit(
            f"❌ Modèle introuvable : {MODEL_FILE}\n"
            "Lancez d'abord : python scripts/lstm/train_lstm_casinolimit.py"
        )
    if not os.path.exists(SCALER_FILE):
        raise SystemExit(
            f"❌ Scaler introuvable : {SCALER_FILE}\n"
            "Lancez d'abord : python scripts/lstm/train_lstm_casinolimit.py"
        )

    model, checkpoint = load_model()
    feature_names = checkpoint["feature_names"]
    feature_groups = checkpoint["feature_groups"]
    window = checkpoint["window"]
    scaler = joblib.load(SCALER_FILE)

    df = load_dataframe(SRC)
    X, lengths, y, teams, group_keys = build_sequences(df)
    # IMPORTANT : le modèle a été entraîné sur des features normalisées
    # (StandardScaler ajusté sur TRAIN, voir train_lstm_casinolimit.py) —
    # on applique ici le même scaler avant d'appeler le modèle/l'explainer.
    X = transform_padded(X, lengths, scaler)

    selected = select_attack_sequences(y, teams, args.n_attacks)
    print(f"\n{len(selected)} séquence(s) attaque sélectionnée(s) (une par équipe).")

    explainer = XNIDSExplainer(
        model=model, feature_names=feature_names, group_dict=feature_groups,
        window=window, device=DEVICE, num_samples=200, delta=0.02,
    )

    techniques, embed_model, embeddings = load_mitre_index()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {"config": {"llm_model": LLM_MODEL, "n_attacks": len(selected)}, "events": []}
    score_dense_top15 = 0
    score_agent2_top = {rank: 0 for rank in range(1, 6)}

    for i, idx in enumerate(selected, start=1):
        print("\n" + "-" * 90)
        print(f"🔍 Séquence {i}/{len(selected)} — équipe {teams[idx]} — index {idx}")

        true_label = str(df.loc[group_keys[idx][-1], "label_technique"])
        result = explainer.explain(X[idx], top_n=args.top_n_features)
        explanation_text = build_top_features_text(result["top_features"])
        print("\n🔎 [XNIDS] TOP FEATURES")
        print("-" * 70)
        print(explanation_text)
        flows = format_flows(df, group_keys[idx])

        print(f"Score xNIDS (probabilité attaque) : {result['original_score']:.4f}")
        print(f"Historique retenu : {result['effective_history']}/{result['n_real']}")

        try:
            behavior_description = run_agent1(flows.to_string(), explanation_text)
        except Exception as e:
            print(f"❌ Erreur Agent 1 : {e}")
            continue
        print("\n📝 [AGENT 1 — BEHAVIOR DESCRIPTION]")
        print(behavior_description)

        dense_candidates = dense_retrieval(behavior_description, techniques, embed_model, embeddings, top_k=25)
        dense_top15_ids = [c["id"] for c in dense_candidates]
        if true_label in dense_top15_ids:
            score_dense_top15 += 1

        try:
            top5 = run_agent2(behavior_description, dense_candidates)
        except Exception as e:
            print(f"❌ Erreur Agent 2 : {e}")
            top5 = []

        print("\n🏆 [AGENT 2 — TOP 5]")
        for c in top5:
            print(f"  {c.get('rank')}. {c.get('technique_id')} — {c.get('technique_name')}")
            if c.get("technique_id") == true_label and c.get("rank") in score_agent2_top:
                score_agent2_top[c.get("rank")] += 1

        report["events"].append({
            "event_index": i, "team": str(teams[idx]), "true_technique": true_label,
            "xnids_score": result["original_score"], "xnids_top_features": explanation_text,
            "behavior_description": behavior_description,
            "dense_top15": dense_top15_ids, "agent2_top5": top5,
        })

    total = len(report["events"])
    report["summary"] = {
        "total_events": total,
        "dense_top15_accuracy": score_dense_top15 / total if total else None,
        "agent2_top_k_accuracy": {k: v / total if total else None for k, v in score_agent2_top.items()},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("📊 RAPPORT FINAL")
    print("=" * 90)
    print(json.dumps(report["summary"], indent=2))
    print(f"\n💾 Rapport complet : {args.output}")


if __name__ == "__main__":
    main()
