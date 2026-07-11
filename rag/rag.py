from rag.retriever import Retriever


class SimpleRAG:

    def __init__(self):

        self.retriever = Retriever()

    def process(self, query):

        retrieved_documents = self.retriever.retrieve(query)

        context = []

        for document in retrieved_documents:

            context.append(document["description"])

        return context