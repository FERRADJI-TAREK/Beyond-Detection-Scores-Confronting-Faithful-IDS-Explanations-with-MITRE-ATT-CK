"""Utilitaires génériques partagés par le pipeline."""

import os
import sys


class HiddenPrints:
    """
    Context manager qui supprime temporairement les impressions sur stdout.

    Utilisé pour masquer les logs verbeux de ``predict_proba`` (pytabkit)
    et de l'explainer SHAP pendant le calcul des explications.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
