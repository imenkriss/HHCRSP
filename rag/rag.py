from rag.retriever import Retriever


class SimpleRAG:

    def __init__(self):

        self.retriever = Retriever()

    def process(self, query):

        retrieved_documents = self.retriever.retrieve(query)

        context = []

        for result in retrieved_documents:

            # retrieve() renvoie {"document": {...}, "score": n}
            context.append(result["document"]["description"])

        return context