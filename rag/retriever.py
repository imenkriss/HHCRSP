from connaissance.base_connaissance import KnowledgeBase


class Retriever:

    def __init__(self):
        """
        Charge la base de connaissances.
        """
        self.knowledge_base = KnowledgeBase()

    def retrieve(self, query):
        """
        Recherche les documents les plus pertinents.
        """

        query = query.lower()

        documents = self.knowledge_base.get_all_documents()

        results = []

        for document in documents:

            # Les contraintes n'ont pas encore de keywords
            if "keywords" not in document:
                continue

            score = 0

            for keyword in document["keywords"]:

                if keyword.lower() in query:

                    score += 1

            if score > 0:

                results.append({

                    "document": document,

                    "score": score

                })

        # Trier du meilleur score au plus faible
        results.sort(

            key=lambda x: x["score"],

            reverse=True

        )

        return results