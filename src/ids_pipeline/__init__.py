"""
ids_pipeline
============

Package Python du pipeline IDS Neuro-Symbolique :
RealMLP + SHAP (regroupement hybride sémantique/corrélation)
+ Agents LLM (synthèse comportementale, RAG dense MITRE ATT&CK, classement final).

Sous-packages :
    preprocessing   -- tri chronologique, feature engineering, construction du dataset focus
    modeling        -- entraînement RealMLP + vectorisation pour l'inférence
    explainability  -- SHAP, regroupement hybride, permutation (validation adversariale)
    agents          -- Agent 1 (comportement), RAG MITRE, Agent 2 (classement)
"""

__version__ = "1.0.0"
