import re
SYSTEM_PROMPT = """
You are an HHCOP reasoning assistant.

Your role is to analyze a home healthcare scenario
using the knowledge retrieved by the RAG system.

You must:
- identify the patient concerned;
- identify the caregiver concerned;
- detect patient priority;
- detect caregiver delay;
- identify relevant HHCOP rules and constraints;
- provide structured information to the multi-agent system.

You must NOT:
- assign a caregiver;
- make the final scheduling decision;
- invent missing information;
- ignore retrieved rules or constraints.

The final assignment will be handled by the
multi-agent system and the optimization module.

Return only a valid JSON object.
"""


USER_PROMPT = """
HHCOP scenario:
{query}

Knowledge retrieved by RAG:
{context}

Analyze the scenario and return:

{
    "patient_id": "...",
    "caregiver_id": "...",
    "urgency": "...",
    "delayed": true,
    "relevant_rules": [],
    "constraints": []
}
"""

class LLMRaisonner:
    """
    Simulation d'un raisonnement LLM (extraction structurée sans appel modèle).
    """

    # Un identifiant valide est une lettre suivie de chiffres : P1, S12...
    PATIENT_PATTERN = re.compile(r"^P\d+$", re.IGNORECASE)
    CAREGIVER_PATTERN = re.compile(r"^S\d+$", re.IGNORECASE)

    URGENCY_TERMS = ("urgent", "urgence", "emergency", "priority")
    DELAY_TERMS = ("delay", "delayed", "late", "retard")

    def __init__(self):
        pass

    def analyze(self, query: str, retrieved_docs: list[str]) -> dict:
        """
        Extrait les entités de la requête et retourne une structure
        exploitable par l'orchestrateur.
        """

        query_lower = query.lower()

        patient_id = None
        caregiver_id = None

        # découpage sur tout caractère non alphanumérique
        tokens = re.split(r"[^A-Za-z0-9]+", query)

        for token in tokens:

            if not token:
                continue

            if patient_id is None and self.PATIENT_PATTERN.match(token):
                patient_id = token.upper()

            elif caregiver_id is None and self.CAREGIVER_PATTERN.match(token):
                caregiver_id = token.upper()

        urgency = any(term in query_lower for term in self.URGENCY_TERMS)
        delayed = any(term in query_lower for term in self.DELAY_TERMS)

        return {
            "query": query,
            "patient_id": patient_id,
            "caregiver_id": caregiver_id,
            "urgency": urgency,
            "delayed": delayed,
            "rag_context": retrieved_docs
        }
