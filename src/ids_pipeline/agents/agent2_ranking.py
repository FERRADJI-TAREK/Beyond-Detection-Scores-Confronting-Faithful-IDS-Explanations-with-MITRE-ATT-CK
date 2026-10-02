"""
Étape [8] du pipeline — Agent 2 : classificateur final.

Analyse les techniques MITRE candidates (issues du RAG dense) au regard de
la description comportementale produite par l'Agent 1, et retourne un
classement Top-5 structuré en JSON.
"""

import json
import re

from ids_pipeline import config


def build_agent2_prompt(behavior_description: str, dense_candidates: list[dict]) -> str:
    candidates_text = "".join(
        f"Candidate {i}\nID: {c['id']}\nName: {c['name']}\nDescription: {c['description']}\n\n"
        for i, c in enumerate(dense_candidates, 1)
    )

    return f"""
You are a cybersecurity analyst. Rank the following 5 candidates using ONLY their descriptions.
Return exactly five candidates in JSON format.
Behavior description:
{behavior_description}
Candidates:
{candidates_text}
Return ONLY valid JSON:
{{ "top_5": [ {{"rank": 1, "technique_id": "TXXXX", "technique_name": "...", "reason": "..."}} ] }}
"""


def run_agent2(
    behavior_description: str,
    dense_candidates: list[dict],
    llm_model: str = config.LLM_MODEL,
) -> list[dict]:
    """Appelle le LLM (Ollama) pour classer les candidats et parse la réponse JSON."""
    from ollama import chat  # import tardif (dépendance optionnelle)

    prompt = build_agent2_prompt(behavior_description, dense_candidates)
    response = chat(model=llm_model, messages=[{"role": "user", "content": prompt}])

    content = response.get("message", {}).get("content", "") if isinstance(response, dict) \
        else response.message.content

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return []

    try:
        return json.loads(match.group(0)).get("top_5", [])
    except json.JSONDecodeError:
        return []
