"""
Étape [5b] du pipeline — Création de l'explainer SHAP et calcul des
valeurs SHAP pour une instance.

Utilise un ``shap.maskers.Partition`` (clustering par corrélation) combiné
à un ``shap.PermutationExplainer``, cohérent avec le regroupement hybride
appliqué en aval (``shap_grouping.py``).
"""

import numpy as np
import pandas as pd
import shap

from ids_pipeline import config
from ids_pipeline.utils import HiddenPrints


def build_correlation_matrix(
    df_source: pd.DataFrame,
    feature_cols: list[str],
    row_to_vector,
    sample_size: int = config.CORRELATION_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcule la matrice de corrélation (valeur absolue) sur un échantillon du dataset."""
    df_sample = df_source.sample(n=min(sample_size, len(df_source)), random_state=random_state)
    X_corr_sample = pd.DataFrame([row_to_vector(row) for _, row in df_sample.iterrows()])
    X_corr_sample = X_corr_sample[feature_cols]
    return X_corr_sample.corr().abs().fillna(0)


def create_explainer(predict_fn, background: pd.DataFrame):
    """Crée l'explainer SHAP Permutation avec masker de partition corrélée."""
    masker = shap.maskers.Partition(background, clustering="correlation")
    return shap.PermutationExplainer(predict_fn, masker)


def compute_shap_values(explainer, X_instance: pd.DataFrame) -> np.ndarray:
    """
    Calcule les valeurs SHAP pour une instance et retourne le vecteur
    correspondant à la classe positive (Attaque), aplati en 1D.
    """
    with HiddenPrints():
        shap_result = explainer(X_instance)

    shap_values_instance = shap_result[0]
    shap_array = np.asarray(shap_values_instance.values)

    if shap_array.ndim == 2:
        shap_array = shap_array[:, -1]
    elif shap_array.ndim > 1:
        shap_array = np.squeeze(shap_array)[:, -1]

    return shap_array
