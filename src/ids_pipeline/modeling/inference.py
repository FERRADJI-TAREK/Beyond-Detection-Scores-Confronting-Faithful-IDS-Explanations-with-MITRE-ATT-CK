"""
Chargement du modèle RealMLP et vectorisation d'une instance brute vers les
59 features attendues, utilisés à la fois pour l'inférence et pour SHAP.
"""

import joblib
import numpy as np
import pandas as pd

from ids_pipeline import config
from ids_pipeline.utils import HiddenPrints


def load_realmlp_model(model_path: str = config.REALMLP_MODEL_FILE):
    """Charge le modèle RealMLP entraîné et affiche un résumé."""
    model = joblib.load(model_path)
    print("✅ Modèle RealMLP binaire chargé avec succès.")
    print(f"✅ Nombre de classes : {len(model.classes_)}")
    return model


def get_feature_columns(model) -> list[str]:
    """Récupère l'ordre exact des colonnes attendu par le modèle chargé."""
    return list(model.x_converter_.fitted_columns)


def prepare_realmlp_instance(row: pd.Series, feature_cols: list[str]) -> pd.Series:
    """
    Convertit une ligne brute de log Zeek (enrichie des features
    comportementales) en un vecteur aligné sur ``feature_cols``
    (one-hot proto/service/conn_state + décomposition de ``history``).
    """
    data = {}

    for feature in config.DIRECT_FEATURES:
        if feature in row.index:
            value = row[feature]
            try:
                value = float(value)
            except Exception:
                value = 0.0
            if pd.isna(value):
                value = 0.0
            data[feature] = value

    proto = str(row.get("proto", "")).strip().lower()
    data["proto_icmp"] = int(proto == "icmp")
    data["proto_tcp"] = int(proto == "tcp")
    data["proto_udp"] = int(proto == "udp")

    service = str(row.get("service", "")).strip().lower()
    for feature in config.SERVICE_FEATURES:
        service_name = feature.replace("service_", "", 1)
        data[feature] = int(service == service_name)

    conn_state = str(row.get("conn_state", "")).strip()
    for feature in config.CONN_STATE_FEATURES:
        state = feature.replace("conn_state_", "", 1)
        data[feature] = int(conn_state == state)

    history = str(row.get("history", ""))
    data["hist_has_A"] = int("A" in history)
    data["hist_has_F"] = int("F" in history)
    data["hist_has_R"] = int("R" in history)
    data["hist_has_S"] = int("S" in history)
    data["hist_has_d"] = int("d" in history)
    data["hist_has_h"] = int("h" in history)
    data["hist_length"] = len(history)
    data["hist_scan_behavior"] = int(
        ("S" in history and "F" not in history) or ("S" in history and "R" not in history)
    )

    result = pd.Series(data)
    result = result.reindex(feature_cols, fill_value=0)
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    return result


def build_predict_fn(model, feature_cols: list[str]):
    """Retourne une fonction ``predict_proba`` compatible avec l'explainer SHAP."""

    def predict_fn(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_cols)
        else:
            X = X.copy()
            if list(X.columns) != feature_cols:
                X = X.reindex(columns=feature_cols, fill_value=0)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        with HiddenPrints():
            return model.predict_proba(X)

    return predict_fn
