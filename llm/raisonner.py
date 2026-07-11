class LLMRaisonner:
    def __init__(self):
        pass

    def analyze(self, query: str, retrieved_docs: list[str]) -> dict:
        """
        Simule le raisonnement du LLM à partir de la requête et du contexte RAG.
        Retourne une structure exploitable par l'orchestrateur.
        """

        query_lower = query.lower()

        patient_id = None
        caregiver_id = None
        urgency = False
        delayed = False

        # extraction simple depuis la requête
        tokens = query.replace(",", " ").split()

        for token in tokens:
            if token.upper().startswith("P"):
                patient_id = token.upper()
            elif token.upper().startswith("S"):
                caregiver_id = token.upper()

        if "urgent" in query_lower or "urgence" in query_lower:
            urgency = True

        if "delay" in query_lower or "retard" in query_lower or "delayed" in query_lower:
            delayed = True

        return {
            "query": query,
            "patient_id": patient_id,
            "caregiver_id": caregiver_id,
            "urgency": urgency,
            "delayed": delayed,
            "rag_context": retrieved_docs
        }