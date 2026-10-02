#!/usr/bin/env python3
"""
Entraînement du LSTM binaire CasinoLimit (séquences de 10 flux bruts,
masquage explicite du padding via ``pack_padded_sequence``), sur les
7 features de la représentation "Baseline" déclarée Table I de l'article
(3 features de trafic log-transformées + 4 indicatrices de protocole
TCP/UDP/ICMP/GRE — voir ``casinolimit_data.SEQ_FEATURES``), puisque le
LSTM consomme les features Baseline sur une fenêtre temporelle
(Table IV : "Flow window").

Hyperparamètres fixés tels que déclarés dans l'article (reproduits à
l'identique, sans early stopping ni sélection de epoch) :
    - window size W = 10 (min. 3 flux réels)     -> casinolimit_data.WINDOW / MIN_SEQ_LEN
    - hidden size   = 64                          -> LSTMBinary (défaut)
    - layers        = 2 (bidirectionnel)          -> LSTMBinary (défaut)
    - dropout       = 0.2                         -> LSTMBinary (défaut)
    - optimizer     = Adam
    - learning rate = 1e-3
    - weight decay  = 1e-5
    - epochs        = 3
    - batch size    = 256

Sauvegarde le modèle entraîné et le scaler associé dans
``models/casinolimit/``.

Usage:
    python scripts/lstm/train_lstm_casinolimit.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casinolimit_data import (
    FEATURE_GROUPS, SEQ_FEATURES, WINDOW, build_sequences, load_dataframe, real_rows, transform_padded,
)
from lstm_common import LSTMBinary, SequenceDataset, evaluate_lstm, find_best_threshold, train_lstm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "casinolimit")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "casinolimit")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

SRC = os.path.join(DATA_DIR, "final_dataset_14techniques_benign_behavioral.csv")

TEST_SIZE = 0.10
VAL_SIZE = 0.15
RANDOM_STATE = 42
N_EPOCHS = 3        # hyperparamètre déclaré dans l'article — voir scripts/lstm/README.md
BATCH_SIZE = 256    # hyperparamètre déclaré dans l'article

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 80)
    print("🧠 LSTM CasinoLimit — entraînement (8 features, hyperparamètres de l'article)")
    print("=" * 80)
    print(f"Device : {DEVICE}")

    df = load_dataframe(SRC)
    X, lengths, y, teams, _group_keys = build_sequences(df)
    print(f"X shape : {X.shape} | équipes : {len(np.unique(teams))}")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss_test.split(X, y, teams))

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    len_trainval, len_test = lengths[trainval_idx], lengths[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    teams_trainval = teams[trainval_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, teams_trainval))

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    len_train, len_val = len_trainval[train_idx], len_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    print(f"TRAIN : {len(y_train):,}  | VAL : {len(y_val):,}  | TEST : {len(y_test):,}")

    scaler = StandardScaler()
    scaler.fit(real_rows(X_train, len_train))

    X_train = transform_padded(X_train, len_train, scaler)
    X_val = transform_padded(X_val, len_val, scaler)
    X_test = transform_padded(X_test, len_test, scaler)

    train_loader = DataLoader(SequenceDataset(X_train, len_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, len_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test, len_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    pos_weight = torch.tensor([n_negative / max(n_positive, 1)], dtype=torch.float32)
    print(f"pos_weight : {pos_weight.item():.4f}")

    model = LSTMBinary(n_features=len(SEQ_FEATURES))  # hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True (défauts)
    model = train_lstm(model, train_loader, DEVICE, N_EPOCHS, pos_weight)

    print("\n--- Seuil par défaut (0.5) ---")
    evaluate_lstm(model, test_loader, DEVICE, threshold=0.5)

    best_threshold = find_best_threshold(model, val_loader, DEVICE)
    print(f"\n--- Seuil optimal {best_threshold:.2f} (choisi sur VAL, appliqué sur TEST) ---")
    final_metrics = evaluate_lstm(model, test_loader, DEVICE, threshold=best_threshold)

    results_file = os.path.join(RESULTS_DIR, "lstm_casinolimit_test_results.csv")
    pd.DataFrame([{"Model": "LSTM", **final_metrics}]).to_csv(results_file, index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": len(SEQ_FEATURES),
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "window": WINDOW,
        "threshold": float(best_threshold),
        "feature_names": SEQ_FEATURES,
        "feature_groups": FEATURE_GROUPS,
    }
    model_file = os.path.join(MODEL_DIR, "lstm_timeseries.pt")
    torch.save(checkpoint, model_file)

    scaler_file = os.path.join(MODEL_DIR, "lstm_timeseries_scaler.joblib")
    joblib.dump(scaler, scaler_file)

    print(f"\n💾 Résultats : {results_file}")
    print(f"💾 Modèle    : {model_file}")
    print(f"💾 Scaler    : {scaler_file}")


if __name__ == "__main__":
    main()
