from __future__ import annotations

from rag.retriever import Retriever
from rag.vector_store import InMemoryVectorStore


class SimpleRAG:
    def __init__(self):
        self.retriever = Retriever()

    def process(self, query: str, top_k: int | None = None) -> list[str]:
        retrieved_documents = self.retriever.retrieve(query, top_k=top_k)
        context = []
        for result in retrieved_documents:
            context.append(result["document"]["description"])
        return context


class RAGPipeline:
    STAGES = [
        "knowledge_base",
        "embeddings",
        "vector_store",
        "retrieval",
        "context_enrichment",
        "reasoning",
    ]

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.retriever = Retriever()
        self.vector_store = InMemoryVectorStore() if embedder is not None else None
        if self.vector_store is not None:
            for document in self.retriever.knowledge_base.get_rules():
                text = document.get("text") or document.get("description") or ""
                self.vector_store.add(document["id"], self.embedder.embed(text))

    def retrieval(self, query: str, top_k: int | None = None) -> list[dict]:
        results = self.retriever.retrieve(query, top_k=top_k)
        if self.vector_store is None or self.embedder is None:
            return results
        query_vector = self.embedder.embed(query)
        vector_matches = self.vector_store.search(query_vector, top_k=top_k)
        hybrid = {doc_id: score for doc_id, score in vector_matches.items() if doc_id in {result["document"]["id"] for result in results}}
        if not hybrid:
            return results
        merged = {result["document"]["id"]: result for result in results}
        for doc_id, similarity in hybrid.items():
            if doc_id in merged:
                merged[doc_id]["score"] = max(merged[doc_id]["score"], float(similarity))
        return sorted(merged.values(), key=lambda item: (-item["score"], item["document"]["id"]))

    def run(self, query: str, top_k: int | None = None) -> dict:
        retrieved = self.retrieval(query, top_k=top_k)
        context = [result["document"]["description"] for result in retrieved]
        return {
            "retrieved": retrieved,
            "context": context,
            "reasoning": None,
        }
