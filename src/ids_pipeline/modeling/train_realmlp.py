"""
Étape [4] du pipeline — Entraînement du modèle RealMLP (pytabkit).

Charge le dataset focus (BENIGN / T1046 / T1595 /T1587), encode les variables
catégorielles (one-hot), aligne les colonnes sur les 59 features attendues
(``config.EXPECTED_FEATURES``), entraîne un ``RealMLP_TD_Classifier``
binaire (Bénin vs Attaque) et sauvegarde le modèle.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ids_pipeline import config

warnings.filterwarnings("ignore")


def load_and_encode_dataset(dataset_path: str = config.FOCUS_DATASET_FILE):
    """Charge le dataset focus, encode les catégorielles et aligne sur les 63 features."""
    print(f"\n📂 Chargement du dataset :\n{dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"✅ Nombre total de flux d'origine : {len(df)}")

    label_col = next((c for c in config.POSSIBLE_LABEL_COLS if c in df.columns), None)
    if label_col is None:
        raise ValueError("❌ Impossible de trouver la colonne de label technique dans le dataset.")

    # Cible binaire : Attaque = 1, Bénin ('none') = 0
    df["target"] = np.where(df[label_col].astype(str).str.lower() == "none", 0, 1)

    feature_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS_FOR_TRAINING]
    df = df.replace("-", np.nan)

    categorical_cols = df[feature_cols].select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).replace("nan", "inconnu")

    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    for feature in df.columns:
        if feature not in config.EXCLUDE_COLS_FOR_TRAINING:
            df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)

    # Alignement exact sur les 63 features attendues
    for col in config.EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    X = df[config.EXPECTED_FEATURES].copy()
    y = df["target"].copy()

    print("=" * 80)
    print(f"✅ RealMLP attend {len(config.EXPECTED_FEATURES)} features")
    print(f"✅ Dataset final généré avec {X.shape[1]} features")
    print("=" * 80)

    return X, y


def train(
    dataset_path: str = config.FOCUS_DATASET_FILE,
    model_output_path: str = config.REALMLP_MODEL_FILE,
    device: str = "cuda",
    random_state: int = 42,
):
    from pytabkit import RealMLP_TD_Classifier  # import tardif (dépendance optionnelle)

    X, y = load_and_encode_dataset(dataset_path)

    df_combined = pd.concat([X, y], axis=1).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    X = df_combined[config.EXPECTED_FEATURES]
    y = df_combined["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    print("\n📚 Dimensions TRAIN :", X_train.shape)
    print("🧪 Dimensions TEST  :", X_test.shape)

    print("\n" + "=" * 80)
    print("⏳ ENTRAÎNEMENT RealMLP EN COURS...")
    print("=" * 80)

    mlp_model = RealMLP_TD_Classifier(device=device, random_state=random_state)
    mlp_model.fit(X_train, y_train)
    print("\n✅ Entraînement terminé.")

    print("\n" + "=" * 80)
    print("🔮 ÉVALUATION")
    print("=" * 80)

    y_pred = mlp_model.predict(X_test)
    print(f"\nAccuracy          : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
    print("\n📋 Classification Report :\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(mlp_model, model_output_path)
    print(f"\n💾 Modèle sauvegardé avec succès dans : {model_output_path}")

    return mlp_model


def main() -> None:
    train()


if __name__ == "__main__":
    main()
