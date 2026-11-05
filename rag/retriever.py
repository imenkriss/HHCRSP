from connaissance.base_connaissance import KnowledgeBase
import re
import unicodedata


class Retriever:

    def __init__(self):
        """
        Charge la base de connaissances.
        """
        self.knowledge_base = KnowledgeBase()

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        without_accents = "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        )
        return re.sub(r"\s+", " ", without_accents.casefold()).strip()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrouve uniquement les règles pertinentes pour la requête."""

        normalized_query = self._normalize(query)
        if not normalized_query:
            return []

        documents = self.knowledge_base.get_all_documents()
        results = []

        for document in documents:
            if "keywords" not in document or not document.get("id", "").startswith(("R", "C")):
                continue

            score = 0
            for keyword in document["keywords"]:
                normalized_keyword = self._normalize(keyword)
                if normalized_keyword and normalized_keyword in normalized_query:
                    score += 1

            if score > 0:
                results.append({"document": document, "score": score})

        has_urgent_signal = bool(re.search(r"\b(?:urgent|emergency|priority|critical)\b", normalized_query))
        has_delay_signal = bool(re.search(r"\b(?:delay|delayed|late|retard|traffic)\b", normalized_query))
        if has_urgent_signal and has_delay_signal:
            results = [item for item in results if item["document"]["id"].startswith("R")]
        elif has_delay_signal:
            results = [item for item in results if item["document"]["id"].startswith(("R", "C"))]
        else:
            results = [item for item in results if item["document"]["id"].startswith("R")]

        results.sort(key=lambda result: (-result["score"], result["document"]["id"]))

        if top_k is not None:
            return results[: max(0, top_k)]

        return results