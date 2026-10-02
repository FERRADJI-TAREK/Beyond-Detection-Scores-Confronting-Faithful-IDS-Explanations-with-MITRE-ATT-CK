"""
Étape [3] du pipeline — Filtrage et fusion vers le dataset final.

Ne conserve que le trafic **BENIGN**, **T1046** (Network Service Scanning)
et **T1595** (Active Scanning) et T1587  à partir des fichiers ``*_sorted.csv``
enrichis, puis fusionne le tout dans un unique CSV prêt pour l'entraînement
et le pipeline SHAP/agents.

Justification (voir section 1.4 du rapport scientifique) : ces trois
classes concentrent l'essentiel des flux disponibles ; les autres
techniques MITRE ATT&CK sont trop peu représentées pour un apprentissage
équilibré.
"""

import glob
import os

import pandas as pd

from ids_pipeline import config

TECHNIQUES_A_GARDER = ["t1595", "t1046",'t1587', "none"]  # 'none' = trafic bénin


def build_focus_dataset(
    raw_dir: str = config.RAW_DATA_DIR,
    output_path: str = config.FOCUS_DATASET_FILE,
) -> pd.DataFrame:
    fichiers_csv = glob.glob(os.path.join(raw_dir, "*_sorted.csv"))
    dataframes_filtres = []

    print("🔍 Début du filtrage et du regroupement...\n")
    for fichier in fichiers_csv:
        print(f"📁 Lecture de {os.path.basename(fichier)}...")
        df = pd.read_csv(fichier)

        mask = df["label_technique"].astype(str).str.lower().isin(TECHNIQUES_A_GARDER)
        df_filtre = df[mask]
        dataframes_filtres.append(df_filtre)
        print(f"   => {len(df_filtre)} lignes conservées sur {len(df)}.")

    print("\n🔄 Fusion des données en cours...")
    df_final = pd.concat(dataframes_filtres, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print("\n✅ OPÉRATION TERMINÉE !")
    print(f"👉 {output_path}")
    print(f"📊 Taille du nouveau dataset : {len(df_final)} lignes.")
    return df_final


def main() -> None:
    build_focus_dataset()


if __name__ == "__main__":
    main()
