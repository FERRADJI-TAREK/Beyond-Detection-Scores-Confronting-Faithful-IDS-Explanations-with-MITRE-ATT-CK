"""
Étape [7] du pipeline — Retrieval dense (RAG) sur la base de connaissances
MITRE ATT&CK.

Encode chaque technique MITRE (ID + nom + description comportementale et
officielle) via un ``SentenceTransformer``, puis retrouve, pour une
description comportementale donnée (sortie de l'Agent 1), les techniques
candidates les plus proches par similarité cosinus.
"""

import json

import numpy as np

from ids_pipeline import config


def load_mitre_techniques(mitre_json_path: str = config.MITRE_JSON_FILE) -> list[dict]:
    """Charge et normalise la base MITRE ATT&CK depuis le fichier JSON."""
    with open(mitre_json_path, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)

    techniques = []
    for tech in mitre_data:
        desc_behav = tech.get("behavioral_description", "")
        desc_mitre = tech.get("mitre_description", "")
        full_desc = f"{desc_behav} {desc_mitre}".strip()

        techniques.append({
            "id": tech.get("technique_id", ""),
            "name": tech.get("name", ""),
            "description": full_desc,
        })

    return techniques


class MitreDenseIndex:
    """Index de recherche dense sur la base MITRE ATT&CK."""

    def __init__(self, techniques: list[dict], embedding_model_name: str = config.EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # import tardif

        self.techniques = techniques
        self.model = SentenceTransformer(embedding_model_name)

        texts = [f"{t['id']} {t['name']} {t['description']}" for t in techniques]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def dense_retrieval(self, behavior_description: str, top_k: int = config.DENSE_RETRIEVAL_TOP_K) -> list[dict]:
        """Retourne les ``top_k`` techniques MITRE les plus proches de la description donnée."""
        query_embedding = self.model.encode(
            [behavior_description], normalize_embeddings=True, show_progress_bar=False
        )[0]

        similarities = np.dot(self.embeddings, query_embedding)
        ranking = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "id": self.techniques[idx]["id"],
                "name": self.techniques[idx]["name"],
                "description": self.techniques[idx]["description"],
                "similarity": float(similarities[idx]),
            }
            for idx in ranking
        ]
