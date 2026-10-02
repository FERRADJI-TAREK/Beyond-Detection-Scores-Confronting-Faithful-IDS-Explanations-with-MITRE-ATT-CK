"""
Étape [1] du pipeline — Tri chronologique des exports Zeek bruts.

Chaque export Spark (``part-*.csv``) est chargé, trié par la colonne ``ts``
(timestamp Unix, tri numérique donc instantané), puis sauvegardé sous
``*_sorted.csv`` à côté de l'original. Cette étape est un prérequis pour
toutes les features temporelles / fenêtres glissantes calculées ensuite
par ``feature_engineering.py``.
"""

import glob
import os

import pandas as pd

from ids_pipeline import config


def find_raw_part_files(raw_dir: str = config.RAW_DATA_DIR) -> list[str]:
    """Retourne la liste des fichiers ``part-*.csv`` non encore triés."""
    all_parts = glob.glob(os.path.join(raw_dir, "part-*.csv"))
    return [f for f in all_parts if "_sorted" not in f]


def sort_flow_file(filepath: str) -> str:
    """
    Trie un fichier CSV de flux Zeek par ordre chronologique (colonne ``ts``).

    Retourne le chemin du fichier trié généré (``<nom>_sorted.csv``).
    """
    df = pd.read_csv(filepath)
    df = df.sort_values(by="ts", ascending=True).reset_index(drop=True)

    sorted_path = filepath.replace(".csv", "_sorted.csv")
    df.to_csv(sorted_path, index=False)
    return sorted_path


def main(raw_dir: str = config.RAW_DATA_DIR) -> list[str]:
    files = find_raw_part_files(raw_dir)
    print(f"🔍 Trouvé {len(files)} fichier(s) à trier dans {raw_dir}\n")

    sorted_files = []
    for filepath in files:
        name = os.path.basename(filepath)
        print(f"⏳ Traitement en cours : {name}...")
        sorted_path = sort_flow_file(filepath)
        sorted_files.append(sorted_path)
        print(f"✅ Terminé ! Sauvegardé sous : {os.path.basename(sorted_path)}\n")

    print("🎉 Le tri de tous les fichiers est terminé !")
    return sorted_files


if __name__ == "__main__":
    main()
