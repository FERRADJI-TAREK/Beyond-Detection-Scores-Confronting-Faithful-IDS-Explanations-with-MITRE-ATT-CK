#!/usr/bin/env python3
"""
Entraînement du LSTM binaire Zeek, sur les 48 features "Baseline"
déclarées Table I de l'article (voir ``zeek_data.BASELINE_FEATURES``),
split par IP source et par technique MITRE (sans chevauchement d'IP
entre train/val/test). Sauvegarde le modèle dans ``models/``.

Hyperparamètres fixés tels que déclarés dans l'article (Table III,
identiques à ``scripts/lstm/train_lstm_casinolimit.py``) : fenêtre 10,
hidden 64, 2 couches bidirectionnelles, dropout 0.2, Adam (lr 1e-3,
weight_decay 1e-5), **3 epochs**, batch 256 — entraînement simple, sans
sélection du meilleur epoch (une version antérieure de ce script
entraînait sur 30 epochs avec sélection du meilleur F1 sur VAL, ce qui
ne correspondait pas à la Table III — voir scripts/lstm/README.md).

Usage:
    python scripts/lstm/train_lstm_zeek.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lstm_common import LSTMBinary, SequenceDataset, evaluate_lstm, find_best_threshold, train_lstm
from zeek_data import SEQ_LEN, load_and_build_sequences, split_by_ip_and_technique

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

SRC = os.path.join(DATA_DIR, "dataset_focus_t1595_t1046_benin.csv")

SEED = 42
BATCH_SIZE = 256   # hyperparamètre déclaré dans l'article (Table III)
N_EPOCHS = 3       # hyperparamètre déclaré dans l'article (Table III)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 80)
    print("🧠 LSTM Zeek — entraînement (48 features Baseline, hyperparamètres de l'article)")
    print("=" * 80)
    print(f"Device : {DEVICE}")

    data = load_and_build_sequences(SRC)
    X_sequences = data["X_sequences"]
    y_sequences = data["y_sequences"]
    seq_src_ip = data["seq_src_ip"]
    seq_technique = data["seq_technique"]
    model_features = data["model_features"]

    print(f"Séquences : {X_sequences.shape} | features : {len(model_features)}")

    train_idx, val_idx, test_idx = split_by_ip_and_technique(seq_src_ip, seq_technique, seed=SEED)
    print(f"Train : {len(train_idx)} | Val : {len(val_idx)} | Test : {len(test_idx)}")
    print("✅ Aucun chevauchement d'IP entre TRAIN / VAL / TEST (vérifié par assertion).")

    lengths_all = np.full(len(X_sequences), SEQ_LEN, dtype=np.int64)

    train_loader = DataLoader(
        SequenceDataset(X_sequences[train_idx], lengths_all[train_idx], y_sequences[train_idx]),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_sequences[val_idx], lengths_all[val_idx], y_sequences[val_idx]),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_sequences[test_idx], lengths_all[test_idx], y_sequences[test_idx]),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    y_train = y_sequences[train_idx]
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()
    pos_weight_value = (n_negative / n_positive) if n_positive > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
    print(f"pos_weight : {pos_weight_value:.4f}")

    model = LSTMBinary(n_features=len(model_features))  # hidden=64, layers=2, dropout=0.2, bidirectional=True (défauts)
    model = train_lstm(model, train_loader, DEVICE, N_EPOCHS, pos_weight)

    print("\n--- Seuil par défaut (0.5) ---")
    evaluate_lstm(model, test_loader, DEVICE, threshold=0.5)

    best_threshold = find_best_threshold(model, val_loader, DEVICE)
    print(f"\n--- Seuil optimal {best_threshold:.2f} (choisi sur VAL, appliqué sur TEST) ---")
    final_metrics = evaluate_lstm(model, test_loader, DEVICE, threshold=best_threshold)

    model_file = os.path.join(MODEL_DIR, "lstm_binary_zeek.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": len(model_features),
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "seq_len": SEQ_LEN,
        "threshold": float(best_threshold),
        "feature_names": model_features,
        "categorical_features": data["categorical_features"],
    }
    torch.save(checkpoint, model_file)

    results_file = os.path.join(RESULTS_DIR, "lstm_zeek_test_results.csv")
    pd.DataFrame([{"Model": "LSTM", **final_metrics}]).to_csv(results_file, index=False)

    print(f"\n💾 Résultats : {results_file}")
    print(f"💾 Modèle    : {model_file}")


if __name__ == "__main__":
    main()
