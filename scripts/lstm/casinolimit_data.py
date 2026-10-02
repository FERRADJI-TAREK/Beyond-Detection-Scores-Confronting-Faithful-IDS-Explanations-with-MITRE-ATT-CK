"""
Construction des séquences LSTM pour CasinoLimit — features brutes
fixes (log_bytes, log_packets, log_duration, protocoles), fenêtre de 10
flux, label = dernier flux de la fenêtre, groupe = (file_name, src_ip).

Feature set aligné sur la représentation "Baseline" déclarée Table I de
l'article (3 volumétriques + 4 indicatrices de protocole TCP/UDP/ICMP/GRE
= 7 features), puisque le LSTM consomme les features Baseline sur une
fenêtre temporelle (Table IV : "Flow window"). Une version antérieure de
ce fichier incluait une 5ᵉ catégorie ``proto_OTHER`` absente de la Table I
— retirée ici (voir ``scripts/lstm/README.md``) ; elle n'avait de toute
façon aucun effet numérique, le dataset ne contenant que TCP/UDP/ICMP/GRE.

Correction notable par rapport à une version antérieure (documentée
dans le notebook source, conservée ici) : le padding est ajouté APRÈS
la séquence réelle (pas avant), et la longueur réelle de chaque
fenêtre est suivie explicitement, pour que ``pack_padded_sequence``
garantisse que le LSTM ignore le padding — une version précédente avec
padding à gauche non masqué laissait fuiter une corrélation entre la
quantité de padding et le label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 10
MIN_SEQ_LEN = 3

WELL_KNOWN_PROTOCOLS = ["TCP", "UDP", "ICMP", "GRE"]

SEQ_FEATURES = (
    ["log_bytes", "log_packets", "log_duration"]
    + [f"proto_{p}" for p in WELL_KNOWN_PROTOCOLS]
)

FEATURE_GROUPS = {
    "trafic": ["log_bytes", "log_packets", "log_duration"],
    "protocole": [f"proto_{p}" for p in WELL_KNOWN_PROTOCOLS],
}


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp_dt"]).copy()

    df["team"] = df["file_name"].astype(str).str.replace(".csv", "", regex=False)
    df["y"] = (df["label_technique"] != "BENIGN").astype(np.int64)

    for p in WELL_KNOWN_PROTOCOLS:
        df[f"proto_{p}"] = (df["protocol"] == p).astype(np.float32)
    df["proto_OTHER"] = (~df["protocol"].isin(WELL_KNOWN_PROTOCOLS)).astype(np.float32)

    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0)
    df["packets"] = pd.to_numeric(df["packets"], errors="coerce").fillna(0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    df["log_bytes"] = np.log1p(df["bytes"].clip(lower=0))
    df["log_packets"] = np.log1p(df["packets"].clip(lower=0))
    df["log_duration"] = np.log1p(df["duration"].clip(lower=0))

    for feature in SEQ_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def build_sequences(df):
    """Groupe par (file_name, src_ip), construit les fenêtres glissantes
    avec padding À DROITE et longueur réelle explicite. Retourne aussi,
    pour chaque séquence, les indices de lignes du DataFrame d'origine
    (``group_keys``) afin de pouvoir réafficher les flux bruts pour
    Agent 1 dans le pipeline xNIDS."""
    X_list, len_list, y_list, teams, group_keys = [], [], [], [], []

    grouped = df.sort_values("timestamp_dt").groupby(["file_name", "src_ip"], sort=False)

    for (_file_name, _src_ip), g in grouped:
        if len(g) < MIN_SEQ_LEN:
            continue

        g = g.sort_values("timestamp_dt")
        features = g[SEQ_FEATURES].to_numpy(dtype=np.float32)
        labels = g["y"].to_numpy(dtype=np.int64)
        team = g["team"].iloc[0]
        row_index = g.index.to_numpy()

        for end in range(MIN_SEQ_LEN - 1, len(g)):
            start = max(0, end - WINDOW + 1)
            seq = features[start:end + 1]
            n_real = len(seq)

            if n_real < WINDOW:
                pad = np.zeros((WINDOW - n_real, len(SEQ_FEATURES)), dtype=np.float32)
                seq = np.vstack([seq, pad])

            X_list.append(seq)
            len_list.append(n_real)
            y_list.append(labels[end])
            teams.append(team)
            group_keys.append(row_index[start:end + 1])

    X = np.stack(X_list)
    lengths = np.array(len_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.int64)
    teams = np.array(teams)
    return X, lengths, y, teams, group_keys


def real_rows(Xarr, lens):
    """Empile uniquement les positions réelles (hors padding) de chaque
    séquence — utilisé pour ajuster le scaler sans jamais regarder le
    padding, qui doit rester à 0."""
    return np.vstack([Xarr[i, :lens[i]] for i in range(len(lens))])


def transform_padded(Xarr, lens, scaler):
    """Applique un scaler déjà ajusté, uniquement sur les positions
    réelles de chaque séquence (le padding reste à 0)."""
    out = Xarr.copy()
    for i in range(len(lens)):
        n = lens[i]
        out[i, :n] = scaler.transform(Xarr[i, :n])
    return out
