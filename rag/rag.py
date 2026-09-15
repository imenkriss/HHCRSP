from rag.retriever import Retriever


class SimpleRAG:

    def __init__(self):

        self.retriever = Retriever()

    def process(self, query: str, top_k: int | None = None) -> list[str]:

        retrieved_documents = self.retriever.retrieve(query, top_k=top_k)

        context = []

        for result in retrieved_documents:

            # retrieve() renvoie {"document": {...}, "score": n}
            context.append(result["document"]["description"])

        return context