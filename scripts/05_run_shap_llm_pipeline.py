#!/usr/bin/env python3
"""
Etapes [5] a [8] : SHAP + regroupement hybride + Agent 1 + RAG dense MITRE + Agent 2.

Usage:
    python scripts/05_run_shap_llm_pipeline.py
    python scripts/05_run_shap_llm_pipeline.py --permute-shap --seed 42
    python scripts/05_run_shap_llm_pipeline.py --fake-shap-constant --fake-shap-variant behavioral
"""

from ids_pipeline.pipeline import main

if __name__ == "__main__":
    main()
