"""
Étape [5a] du pipeline — Regroupement hybride des features
(familles sémantiques métier ⊂ corrélation Pearson) et agrégation des
valeurs SHAP par groupe.

Approche :
  1. Les features sont d'abord réparties dans des "boîtes" métier
     (``config.SEMANTIC_BUCKETS``) : volume de trafic, dynamique du flux,
     activité source, diversité des cibles, comportement de scan, etc.
  2. À l'intérieur de chaque boîte, un DFS regroupe les features
     corrélées (|r| >= ``config.CORRELATION_THRESHOLD``) en un même
     "groupe évidentiel", pour éviter de compter plusieurs fois la même
     information redondante.
  3. Les valeurs SHAP signées/absolues sont ensuite sommées par groupe et
     triées par importance absolue décroissante.

Ce module contient également ``permute_grouped_results``, utilisée pour la
**validation adversariale** (mélange du classement / des scores / des
signes SHAP) décrite en section 4.2 du rapport scientifique.
"""

import random

import numpy as np
import pandas as pd

from ids_pipeline import config


def build_feature_groups(
    corr_matrix: pd.DataFrame,
    feature_cols: list[str],
    semantic_buckets: dict[str, list[str]] = config.SEMANTIC_BUCKETS,
    correlation_threshold: float = config.CORRELATION_THRESHOLD,
) -> list[dict]:
    """
    Construit les groupes hybrides (sémantique + corrélation) une fois pour
    toutes, à partir d'une matrice de corrélation calculée sur un échantillon
    représentatif du dataset.
    """
    assigned_features = set(f for bucket in semantic_buckets.values() for f in bucket)
    other_features = [f for f in feature_cols if f not in assigned_features]
    buckets = dict(semantic_buckets)
    if other_features:
        buckets["Other_Features"] = other_features

    feature_groups = []
    for bucket_name, bucket_features in buckets.items():
        available_features = [f for f in bucket_features if f in feature_cols]

        visited = set()
        for feature in available_features:
            if feature in visited:
                continue

            group = set()
            stack = [feature]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                group.add(current)

                correlated = corr_matrix.loc[current][
                    corr_matrix.loc[current] >= correlation_threshold
                ].index.tolist()
                for corr_feature in correlated:
                    if corr_feature in available_features and corr_feature not in visited:
                        stack.append(corr_feature)

            feature_groups.append({"semantic_name": bucket_name, "features": sorted(group)})

    return feature_groups


def aggregate_shap_by_correlation(
    shap_values_instance: np.ndarray,
    feature_names: list[str],
    feature_groups: list[dict],
) -> list[dict]:
    """Agrège les valeurs SHAP d'une instance par groupe hybride, triées par |SHAP| décroissant."""
    grouped_results = []

    for item in feature_groups:
        indices = [feature_names.index(f) for f in item["features"]]
        values = np.asarray(shap_values_instance)[indices]

        total_signed = float(np.sum(values))
        total_abs = float(np.sum(np.abs(values)))
        sign = "+" if total_signed > 0 else ("-" if total_signed < 0 else "0")

        grouped_results.append({
            "semantic_name": item["semantic_name"],
            "features": item["features"],
            "signed_shap": total_signed,
            "abs_shap": total_abs,
            "sign": sign,
        })

    grouped_results.sort(key=lambda x: x["abs_shap"], reverse=True)
    return grouped_results


def permute_grouped_results(grouped_results: list[dict], rng: random.Random) -> list[dict]:
    """
    Mélange aléatoirement (signed_shap, abs_shap, sign) ENTRE les groupes,
    sans changer la composition des groupes (semantic_name / features).

    But : injecter une "fausse" évidence SHAP (même structure, valeurs
    scramblées) pour mesurer l'impact réel de SHAP sur la décision des
    agents LLM — validation adversariale, section 4.2 du rapport.
    """
    permuted = [dict(item) for item in grouped_results]

    triplets = [(item["signed_shap"], item["abs_shap"], item["sign"]) for item in permuted]
    shuffled = triplets.copy()
    rng.shuffle(shuffled)

    for item, (signed, absval, sign) in zip(permuted, shuffled):
        item["signed_shap"] = signed
        item["abs_shap"] = absval
        item["sign"] = sign

    permuted.sort(key=lambda x: x["abs_shap"], reverse=True)
    return permuted


def build_explanation_text(grouped_results: list[dict], max_groups: int = 12) -> str:
    """Construit le texte d'évidence SHAP transmis à l'Agent 1 (prompt LLM)."""
    selected = grouped_results[:max_groups]
    lines = []

    for rank, item in enumerate(selected, start=1):
        group_features = item["features"]
        semantic_name = item["semantic_name"]

        descriptions = [config.FEATURE_DESCRIPTIONS.get(f, f) for f in group_features]

        if len(group_features) == 1:
            feature_label = f"[{semantic_name}] {group_features[0]}"
        else:
            feature_label = f"[{semantic_name}] Correlated group: " + ", ".join(group_features)

        description_text = " / ".join(descriptions)
        lines.append(
            f"Rank {rank} | [{item['sign']}] | {feature_label} | Behavioral meaning: {description_text}"
        )

    return "\n".join(lines)
