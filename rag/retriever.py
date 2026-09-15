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
        """
        Recherche les documents les plus pertinents.
        """

        normalized_query = self._normalize(query)

        documents = self.knowledge_base.get_all_documents()

        results = []

        for document in documents:

            # Les contraintes n'ont pas encore de keywords
            if "keywords" not in document:
                continue

            score = 0

            for keyword in document["keywords"]:

                normalized_keyword = self._normalize(keyword)
                if normalized_keyword and normalized_keyword in normalized_query:

                    score += 1

            if score > 0:

                results.append({

                    "document": document,

                    "score": score

                })

        # Trier du meilleur score au plus faible
        results.sort(key=lambda result: (-result["score"], result["document"]["id"]))

        if top_k is not None:
            return results[:max(0, top_k)]

        return results