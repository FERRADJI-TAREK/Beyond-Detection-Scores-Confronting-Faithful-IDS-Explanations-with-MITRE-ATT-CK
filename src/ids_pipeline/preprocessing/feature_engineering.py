"""
Étape [2] du pipeline — Feature engineering comportemental avancé.

Transforme les logs Zeek triés chronologiquement (``*_sorted.csv``) en une
matrice comportementale riche, en calculant pour chaque IP source/destination
des métriques temporelles, de vélocité, d'unicité des cibles/ports et de
taux d'échec de connexion — voir section 1 du rapport scientifique
(``docs/RAPPORT_SCIENTIFIQUE.md``).

Le fichier est enrichi et **écrasé en place** (mêmes conventions que le
notebook d'exploration original).
"""

import glob
import os

import numpy as np
import pandas as pd

from ids_pipeline import config

ROLLING_WINDOW = 10


def ajouter_features_comportementales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute au DataFrame de flux Zeek l'ensemble des features comportementales
    utilisées par le modèle RealMLP :

    1. Features temporelles (temps depuis le dernier événement source/dest)
    2. Fenêtres glissantes classiques (comptes, sommes sur les 10 derniers événements)
    3. Vélocité et débit (paquets/s, octets/s, taille moyenne de paquet)
    4. Exploration et unicité des cibles/ports (indicateur de scan/sweep)
    5. Taux d'échec de connexion (indicateur de reconnaissance à l'aveugle)
    6. Connexions répétées / nouvelles destinations
    """
    df = df.copy()

    # --- Préparation : forcer le typage numérique ---
    cols_numeriques = ["orig_pkts", "orig_bytes", "duration"]
    for col in cols_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- 1. Features temporelles classiques ---
    df["time_since_previous_src_event"] = df.groupby("src_ip_zeek")["ts"].diff().fillna(0)
    df["time_since_previous_dst_event"] = df.groupby("dest_ip_zeek")["ts"].diff().fillna(0)

    # --- 2. Rolling windows classiques (sommes et comptes) ---
    df["src_event_count_last_10"] = df.groupby("src_ip_zeek")["ts"].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).count()
    )
    df["src_packets_sum_last_10"] = df.groupby("src_ip_zeek")["orig_pkts"].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()
    )
    df["src_bytes_sum_last_10"] = df.groupby("src_ip_zeek")["orig_bytes"].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()
    )
    df["src_duration_sum_last_10"] = df.groupby("src_ip_zeek")["duration"].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()
    )

    # --- 3. Vélocité et débit ---
    duration_safe = df["duration"].replace(0, 1e-6)
    pkts_safe = df["orig_pkts"].replace(0, 1)

    df["pkts_per_second"] = df["orig_pkts"] / duration_safe
    df["bytes_per_second"] = df["orig_bytes"] / duration_safe
    df["avg_bytes_per_packet"] = df["orig_bytes"] / pkts_safe

    # --- 4. Exploration et unicité des cibles / ports ---
    df["dest_ip_int"] = pd.factorize(df["dest_ip_zeek"])[0]
    df["src_ip_int"] = pd.factorize(df["src_ip_zeek"])[0]

    df["src_unique_dst_last_10"] = df.groupby("src_ip_zeek")["dest_ip_int"].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).apply(
            lambda ips: len(np.unique(ips)), raw=True
        )
    )
    df["dst_unique_src_last_10"] = df.groupby("dest_ip_zeek")["src_ip_int"].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).apply(
            lambda ips: len(np.unique(ips)), raw=True
        )
    )
    df = df.drop(columns=["dest_ip_int", "src_ip_int"])

    col_port = "dest_port_zeek" if "dest_port_zeek" in df.columns else (
        "id.resp_p" if "id.resp_p" in df.columns else None
    )
    if col_port:
        df["dest_port_int"] = pd.factorize(df[col_port])[0]
        df["src_unique_port_last_10"] = df.groupby("src_ip_zeek")["dest_port_int"].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).apply(
                lambda ports: len(np.unique(ports)), raw=True
            )
        )
        df = df.drop(columns=["dest_port_int"])

    # --- 5. Taux d'échec de connexion (scanners "à l'aveugle") ---
    col_state = "conn_state_zeek" if "conn_state_zeek" in df.columns else (
        "conn_state" if "conn_state" in df.columns else None
    )
    if col_state:
        df["is_failed_conn"] = df[col_state].isin(["S0", "REJ", "RSTOS0"]).astype(int)
        df["src_failed_conn_last_10"] = df.groupby("src_ip_zeek")["is_failed_conn"].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()
        )
        df = df.drop(columns=["is_failed_conn"])

    # --- 6. Connexions répétées / nouvelles destinations ---
    df["repeated_connection"] = df.duplicated(
        subset=["src_ip_zeek", "dest_ip_zeek"], keep="first"
    ).astype(int)
    df["new_destination"] = (~df.duplicated(subset=["dest_ip_zeek"], keep="first")).astype(int)

    return df


def main(raw_dir: str = config.RAW_DATA_DIR) -> None:
    fichiers_csv = glob.glob(os.path.join(raw_dir, "*_sorted.csv"))
    print(f"🔍 {len(fichiers_csv)} fichier(s) à enrichir.\n")

    for fichier in fichiers_csv:
        print(f"📁 Ouverture de {os.path.basename(fichier)}")
        df = pd.read_csv(fichier)

        print("   ⏳ Ajout des features comportementales avancées...")
        df_enrichi = ajouter_features_comportementales(df)

        df_enrichi.to_csv(fichier, index=False)
        print(f"✅ Remplacement terminé pour : {os.path.basename(fichier)}\n")

    print("🎉 Toutes les données sont enrichies avec les features comportementales !")


if __name__ == "__main__":
    main()
