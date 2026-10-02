#!/usr/bin/env python3
"""
Pipeline complet Zeek : LSTM (déjà entraîné) -> xNIDS (Sparse Group
Lasso, auto-contenu, méthode identique à CasinoLimit) -> Agent 1 (LLM)
-> RAG dense MITRE ATT&CK -> Agent 2 (LLM, Top-5).

Remplace la version précédente, qui dépendait de fichiers précalculés
(``instances.npz`` / ``xnids_scores.csv`` / ``xnids_meta.csv``, dont le
générateur n'était pas fourni) : l'explicabilité xNIDS est ici calculée
dynamiquement, à partir du modèle LSTM seul — voir
``scripts/lstm/README.md``, section "Choix d'architecture".

Pré-requis : avoir lancé ``scripts/lstm/train_lstm_zeek.py`` (modèle
non fourni dans cette archive, voir ``models/README.md``).

Usage:
    python scripts/lstm/run_lstm_pipeline_zeek.py --n-per-technique 5
"""

import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lstm_common import LSTMBinary, XNIDSExplainer, build_feature_groups_from_onehot, build_top_features_text
from zeek_data import SEQ_LEN, load_and_build_sequences, split_by_ip_and_technique

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
MITRE_DIR = os.path.join(REPO_ROOT, "mitre")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

SRC = os.path.join(DATA_DIR, "dataset_focus_t1595_t1046_t1587_benin.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_binary_zeek.pt")
MITRE_JSON = os.path.join(MITRE_DIR, "mitre_techniques_behavioral.json")

LLM_MODEL = os.environ.get("IDS_LLM_MODEL", "gpt-oss:120b-cloud")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_DISPLAY_COLS = [
    "ts", "src_ip_zeek", "dest_ip_zeek", "conn_state", "proto", "service",
    "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
]

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


def select_attack_sequences(y_sequences, seq_technique, test_idx, n_per_technique):
    attack_test_idx = [i for i in test_idx if y_sequences[i] == 1]
    by_technique = {}
    for i in attack_test_idx:
        by_technique.setdefault(str(seq_technique[i]), []).append(i)

    selected = []
    for _technique, idx_list in sorted(by_technique.items()):
        selected.extend(idx_list[:n_per_technique])
    return selected


def format_flows(df_raw, row_idx, cols):
    available = [c for c in cols if c in df_raw.columns]
    flows = df_raw.loc[row_idx, available].reset_index(drop=True)
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
    parser.add_argument("--n-per-technique", type=int, default=5,
                         help="Nombre de séquences attaque testées par technique (pour limiter "
                              "les appels LLM). Remplacez ce 5 par le nombre souhaité.")
    parser.add_argument("--top-n-features", type=int, default=12)
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "lstm_zeek_run.json"))
    args = parser.parse_args()

    LLM_MODEL = args.llm_model

    print("=" * 90)
    print("🚀 Pipeline LSTM Zeek — xNIDS + Agent 1 + RAG dense + Agent 2")
    print("=" * 90)

    if not os.path.exists(MODEL_FILE):
        raise SystemExit(
            f"❌ Modèle introuvable : {MODEL_FILE}\n"
            "Lancez d'abord : python scripts/lstm/train_lstm_zeek.py"
        )

    model, checkpoint = load_model()
    feature_names = checkpoint["feature_names"]
    categorical_features = checkpoint.get("categorical_features", [])
    feature_groups = build_feature_groups_from_onehot(feature_names, categorical_features)
    window = checkpoint.get("seq_len", SEQ_LEN)

    data = load_and_build_sequences(SRC)
    X_sequences = data["X_sequences"]
    y_sequences = data["y_sequences"]
    seq_src_ip = data["seq_src_ip"]
    seq_technique = data["seq_technique"]
    seq_row_idx = data["seq_row_idx"]
    df_raw = data["df_raw"]

    _train_idx, _val_idx, test_idx = split_by_ip_and_technique(seq_src_ip, seq_technique, seed=42)

    selected = select_attack_sequences(y_sequences, seq_technique, test_idx, args.n_per_technique)
    print(f"\n{len(selected)} séquence(s) attaque sélectionnée(s) sur l'ensemble TEST.")

    explainer = XNIDSExplainer(
        model=model, feature_names=feature_names, group_dict=feature_groups,
        window=window, device=DEVICE, num_samples=200, delta=0.02,
    )

    techniques, embed_model, embeddings = load_mitre_index()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {"config": {"llm_model": LLM_MODEL, "n_events": len(selected)}, "events": []}
    score_dense_top15 = 0
    score_agent2_top = {rank: 0 for rank in range(1, 6)}

    for i, idx in enumerate(selected, start=1):
        true_label = str(seq_technique[idx])
        print("\n" + "-" * 90)
        print(f"🔍 Séquence {i}/{len(selected)} — IP {seq_src_ip[idx]} — technique réelle {true_label}")

        result = explainer.explain(X_sequences[idx], top_n=args.top_n_features)
        explanation_text = build_top_features_text(result["top_features"])
        print("\n🔎 [XNIDS] TOP FEATURES")
        print("-" * 70)
        print(explanation_text)
        flows = format_flows(df_raw, seq_row_idx[idx], RAW_DISPLAY_COLS)

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
            "event_index": i, "src_ip": str(seq_src_ip[idx]), "true_technique": true_label,
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
