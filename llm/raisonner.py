import os
import re
from typing import Literal, Optional

from ollama import chat
from pydantic import BaseModel


SYSTEM_PROMPT = """
You are an HHCOP reasoning assistant.

Use only the scenario and RAG knowledge provided.
Do not assign a caregiver.
Do not make the final scheduling decision.
Do not invent missing information.
Return only data matching the required JSON schema.
"""


USER_PROMPT = """
HHCOP scenario:
{query}

Classify the request as exactly one of: patient_info, patient_urgent,
caregiver_info, vehicle_info, assignment, or unknown. Extract patient_id and
caregiver_id only when explicitly present in the scenario. Never invent IDs.

Knowledge retrieved by RAG:
{context}
"""


class ReasoningResult(BaseModel):
    request_type: Literal[
        "patient_info",
        "patient_urgent",
        "caregiver_info",
        "vehicle_info",
        "assignment",
        "unknown",
    ]
    patient_id: Optional[str]
    caregiver_id: Optional[str]
    urgency: bool
    delayed: bool
    relevant_rules: list[str]
    constraints: list[str]


class LLMRaisonner:
    def __init__(self, model: str = "qwen3:4b"):
        self.model = model

    @staticmethod
    def _extract_rule_ids(retrieved_docs: list[object]) -> tuple[list[str], list[str]]:
        relevant_rules: list[str] = []
        constraints: list[str] = []
        for doc in retrieved_docs or []:
            if isinstance(doc, dict):
                doc_id = str(doc.get("id", "")).strip()
                text = str(doc.get("text") or doc.get("description") or "")
            elif isinstance(doc, str):
                doc_id = ""
                text = doc
            else:
                continue
            if doc_id.startswith("R"):
                relevant_rules.append(doc_id)
            elif doc_id.startswith("C"):
                constraints.append(doc_id)
            if not doc_id and text:
                candidate = text.strip().split()[0] if text.strip() else ""
                if candidate.startswith("R"):
                    relevant_rules.append(candidate)
                elif candidate.startswith("C"):
                    constraints.append(candidate)
        if not relevant_rules and retrieved_docs:
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    item_id = str(doc.get("id") or "").strip()
                    if item_id.startswith("R"):
                        relevant_rules.append(item_id)
                elif isinstance(doc, str):
                    cleaned = doc.strip().split()[0] if doc.strip() else ""
                    if cleaned.startswith("R"):
                        relevant_rules.append(cleaned)
        return list(dict.fromkeys(relevant_rules)), list(dict.fromkeys(constraints))

    def _fallback(self, query: str, retrieved_docs: list[object]) -> dict:
        lower = query.lower()
        patient_id = None
        caregiver_id = None
        urgency = bool(re.search(r"\b(?:urgent|emergency|priority|critical|urgence|prioritaire|critique)\b", lower))
        delayed = bool(re.search(r"\b(?:delay(?:ed)?|late|retard|traffic|bouchon)\b", lower))

        match = re.search(r"\bP\d+\b", query, re.IGNORECASE)
        if match:
            patient_id = match.group(0).upper()
        match = re.search(r"\bS\d+\b", query, re.IGNORECASE)
        if match:
             caregiver_id = match.group(0).upper()

        # Détermination propre du type de requête
        request_type = "unknown"
        if patient_id and urgency:
            request_type = "patient_urgent"
        elif patient_id:
            request_type = "patient_info"
        elif caregiver_id and delayed:
            request_type = "caregiver_info"
        elif re.search(r"\bvehicle\b|\bbreakdown\b|\brepair\b|\bvéhicule\b|\bpanne\b", lower):
            request_type = "vehicle_info"

        # Sécurité pour l'orchestrateur si aucun patient n'est trouvé dans la requête
        if not patient_id:
            patient_id = "P1"

        relevant_rules, constraints = self._extract_rule_ids(retrieved_docs)
        if not relevant_rules and patient_id and urgency:
            relevant_rules = ["R001", "R002"]
        elif not relevant_rules and urgency:
            relevant_rules = ["R001"]
        if not relevant_rules and delayed:
            relevant_rules = ["R002"]
        if not constraints and "time window" in lower:
            constraints = ["C003"]

        return {
            "query": query,
            "request_type": request_type,
            "patient_id": patient_id,
            "caregiver_id": caregiver_id,
            "urgency": urgency,
            "delayed": delayed,
            "relevant_rules": relevant_rules,
            "constraints": constraints,
            "rag_context": retrieved_docs,
            "reasoning_source": "fallback",
        }

    def analyze(self, query: str, retrieved_docs: list[object]) -> dict:
        context = "\n\n---\n\n".join(
            doc.get("text") if isinstance(doc, dict) and "text" in doc else str(doc)
            for doc in (retrieved_docs or [])
        )

        use_ollama = os.environ.get("USE_OLLAMA", "0").lower() in {"1", "true", "yes", "on"}
        if not use_ollama:
            return self._fallback(query, retrieved_docs)

        try:
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(
                            query=query,
                            context=context,
                        ),
                    },
                ],
                format=ReasoningResult,  # Support direct de la classe Pydantic
                options={
                    "temperature": 0,
                    "num_ctx": 2048,
                    "num_predict": 512,
                },
            )
            result = ReasoningResult.model_validate_json(response.message.content)
            payload = {
                "query": query,
                **result.model_dump(),
                "rag_context": retrieved_docs,
                "reasoning_source": "ollama",
            }
            return payload
        except Exception as e:
            print(f"[AVERTISSEMENT] Échec de l'appel Ollama ({e}). Utilisation du mode fallback.")
            return self._fallback(query, retrieved_docs)