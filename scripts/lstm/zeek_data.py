"""
Construction des séquences LSTM pour Zeek — exactement les 48 features
"Baseline" déclarées Table I de l'article (8 volumétriques + 3 protocole
one-hot tcp/udp/icmp + 11 états de connexion one-hot + 18 services
one-hot + 8 indicateurs d'historique TCP), fenêtre fixe de 10 événements
par src_ip, split train/val/test par IP source ET par technique MITRE
(aucune IP partagée entre les trois ensembles). Le LSTM consomme la
représentation Baseline sur une fenêtre temporelle (Table IV : "Flow
window"), pas les 63 features Behavioral utilisées par RealMLP/CatBoost
Baseline+BF.

Construction identique à celle utilisée pour RealMLP/CatBoost
(``src/ids_pipeline/modeling/inference.prepare_realmlp_instance`` et
``src/ids_pipeline/config`` pour les listes de référence CONN_STATE_FEATURES/
SERVICE_FEATURES/DIRECT_FEATURES), de façon à ce que les trois détecteurs
voient rigoureusement la même information sur ce dataset.

NOTE MÉTHODOLOGIQUE (conservée telle que fournie, documentée aussi dans
OPEN_SCIENCE.md) : le ``StandardScaler`` ci-dessous est ajusté (fit) sur
l'ensemble du dataset AVANT le split train/val/test, contrairement à
CasinoLimit (scaler ajusté uniquement sur TRAIN, après le split). C'est
une fuite mineure (les statistiques globales du scaler "voient"
indirectement val/test) héritée du notebook source ; elle est
documentée plutôt que corrigée silencieusement, pour ne pas modifier à
l'insu de l'auteur une méthodologie déjà utilisée pour produire des
résultats.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ids_pipeline import config  # noqa: E402

SEQ_LEN = 10

# 8 features volumétriques (Table I) — sous-ensemble non comportemental de
# DIRECT_FEATURES.
VOLUMETRIC_FEATURES = [
    "duration", "missed_bytes", "orig_bytes", "orig_ip_bytes",
    "orig_pkts", "resp_bytes", "resp_ip_bytes", "resp_pkts",
]

PROTO_FEATURES = ["proto_tcp", "proto_udp", "proto_icmp"]  # 3, Table I (Zeek)

HIST_FEATURES = [
    "hist_has_A", "hist_has_F", "hist_has_R", "hist_has_S", "hist_has_d",
    "hist_has_h", "hist_length", "hist_scan_behavior",
]  # 8, Table I

# 8 + 3 + 11 (CONN_STATE_FEATURES) + 18 (SERVICE_FEATURES) + 8 = 48
BASELINE_FEATURES = (
    VOLUMETRIC_FEATURES + PROTO_FEATURES + config.CONN_STATE_FEATURES
    + config.SERVICE_FEATURES + HIST_FEATURES
)

CATEGORICAL_PREFIXES = ["proto", "conn_state", "service", "hist"]


def _build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les 48 colonnes Baseline, vectorisé, pour tout le
    DataFrame — logique identique à
    ``ids_pipeline.modeling.inference.prepare_realmlp_instance``
    (appliquée ligne à ligne pour l'inférence SHAP), restreinte aux
    features non comportementales."""
    out = pd.DataFrame(index=df.index)

    for col in VOLUMETRIC_FEATURES:
        out[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    proto = df.get("proto", "").astype(str).str.strip().str.lower()
    out["proto_tcp"] = (proto == "tcp").astype(int)
    out["proto_udp"] = (proto == "udp").astype(int)
    out["proto_icmp"] = (proto == "icmp").astype(int)

    service = df.get("service", "").astype(str).str.strip().str.lower()
    for feature in config.SERVICE_FEATURES:
        service_name = feature.replace("service_", "", 1)
        out[feature] = (service == service_name).astype(int)

    conn_state = df.get("conn_state", "").astype(str).str.strip()
    for feature in config.CONN_STATE_FEATURES:
        state = feature.replace("conn_state_", "", 1)
        out[feature] = (conn_state == state).astype(int)

    history = df.get("history", "").astype(str)
    out["hist_has_A"] = history.str.contains("A", regex=False).astype(int)
    out["hist_has_F"] = history.str.contains("F", regex=False).astype(int)
    out["hist_has_R"] = history.str.contains("R", regex=False).astype(int)
    out["hist_has_S"] = history.str.contains("S", regex=False).astype(int)
    out["hist_has_d"] = history.str.contains("d", regex=False).astype(int)
    out["hist_has_h"] = history.str.contains("h", regex=False).astype(int)
    out["hist_length"] = history.str.len()
    has_s = history.str.contains("S", regex=False)
    has_f = history.str.contains("F", regex=False)
    has_r = history.str.contains("R", regex=False)
    out["hist_scan_behavior"] = ((has_s & ~has_f) | (has_s & ~has_r)).astype(int)

    return out[BASELINE_FEATURES]


def load_and_build_sequences(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.sort_values(["src_ip_zeek", "ts"]).reset_index(drop=True)

    model_features = list(BASELINE_FEATURES)
    df_features = _build_baseline_features(df)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_raw = df_features[model_features].to_numpy(dtype=np.float32)
    y_raw = df["label_binary"].astype(int).to_numpy()

    # NOTE : fit sur l'ensemble du dataset (voir docstring du module) —
    # comportement hérité du notebook source, documenté dans OPEN_SCIENCE.md.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    X_sequences, y_sequences = [], []
    seq_src_ip, seq_technique, seq_row_idx = [], [], []

    groups = df.groupby("src_ip_zeek").indices
    for src_ip, indices in groups.items():
        indices = np.asarray(indices)
        if len(indices) < SEQ_LEN:
            continue
        for i in range(SEQ_LEN - 1, len(indices)):
            window = indices[i - SEQ_LEN + 1: i + 1]
            X_sequences.append(X_scaled[window])
            y_sequences.append(y_raw[window[-1]])
            seq_src_ip.append(src_ip)
            seq_technique.append(df["label_technique"].iloc[window[-1]])
            seq_row_idx.append(df.index.to_numpy()[window])

    X_sequences = np.asarray(X_sequences, dtype=np.float32)
    y_sequences = np.asarray(y_sequences, dtype=np.float32)
    seq_src_ip = np.asarray(seq_src_ip)
    seq_technique = np.asarray(seq_technique)

    return {
        "X_sequences": X_sequences,
        "y_sequences": y_sequences,
        "seq_src_ip": seq_src_ip,
        "seq_technique": seq_technique,
        "seq_row_idx": seq_row_idx,
        "model_features": model_features,
        "categorical_features": CATEGORICAL_PREFIXES,
        "df_raw": df,
        "df_lstm": df_features,
    }


def assign_ips_to_splits(ip_list, seed, val_frac=0.15, test_frac=0.15):
    rng = np.random.default_rng(seed)
    ips = list(ip_list)
    rng.shuffle(ips)
    n = len(ips)

    if n == 1:
        return {ips[0]: "train"}
    if n == 2:
        return {ips[0]: "train", ips[1]: "test"}
    if n == 3:
        return {ips[0]: "train", ips[1]: "val", ips[2]: "test"}

    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(0, n - n_train - n_test)

    split = {}
    for ip in ips[:n_train]:
        split[ip] = "train"
    for ip in ips[n_train:n_train + n_val]:
        split[ip] = "val"
    for ip in ips[n_train + n_val:]:
        split[ip] = "test"
    return split


def split_by_ip_and_technique(seq_src_ip, seq_technique, seed=42):
    """Assigne chaque IP à train/val/test, technique par technique, pour
    garantir qu'aucune IP n'est partagée entre les trois ensembles."""
    ip_technique_map = {}
    for ip, tech in zip(seq_src_ip, seq_technique):
        ip_technique_map.setdefault(ip, tech)

    techniques = sorted(set(ip_technique_map.values()))
    ip_to_split = {}
    for tech in techniques:
        ips_this_tech = [ip for ip, t in ip_technique_map.items() if t == tech]
        split_map = assign_ips_to_splits(ips_this_tech, seed=seed)
        ip_to_split.update(split_map)

    split_labels = np.array([ip_to_split.get(ip, "train") for ip in seq_src_ip])
    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]

    ip_train = set(seq_src_ip[train_idx])
    ip_val = set(seq_src_ip[val_idx])
    ip_test = set(seq_src_ip[test_idx])
    assert len(ip_train & ip_test) == 0, "Fuite IP train/test"
    assert len(ip_train & ip_val) == 0, "Fuite IP train/val"
    assert len(ip_val & ip_test) == 0, "Fuite IP val/test"

    return train_idx, val_idx, test_idx
