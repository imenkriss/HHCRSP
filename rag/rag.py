from __future__ import annotations

from typing import Any, Iterable

from rag.retriever import Retriever
from rag.vector_store import InMemoryVectorStore

DEFAULT_CHUNK_CHARS = 400
DEFAULT_CHUNK_OVERLAP = 50


def chunk_document(
    document: dict[str, Any],
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split one knowledge document into overlapping chunks.

    Each chunk keeps a reference to its parent document id and category, so a
    retrieval result can always be traced back to the original rule. Knowledge
    base rules are short, so in practice one document yields one chunk; the
    function still exists (and is tested) so longer documents are handled
    without silently dropping text.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")
    document_id = str(document.get("id", ""))
    text = str(document.get("text") or document.get("description") or "").strip()
    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append({
            "chunk_id": f"{document_id}#c{index}",
            "document_id": document_id,
            "chunk_index": index,
            "text": text[start:end],
            "category": document.get("category"),
            "document": document,
        })
        if end >= len(text):
            break
        start = end - overlap
        index += 1
    return chunks


def chunk_documents(
    documents: Iterable[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk a document collection, preserving document metadata."""
    chunks: list[dict[str, Any]] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars, overlap=overlap))
    return chunks


class SimpleRAG:
    """Keyword-only RAG facade kept for backward compatibility."""

    def __init__(self):
        self.retriever = Retriever()

    def process(self, query: str, top_k: int | None = None) -> list[str]:
        retrieved_documents = self.retriever.retrieve(query, top_k=top_k)
        context = []
        for result in retrieved_documents:
            context.append(result["document"]["description"])
        return context


class RAGPipeline:
    """Six-stage RAG pipeline (R1).

    Stages: knowledge base -> embeddings -> vector store -> retrieval ->
    context enrichment -> reasoning. Embeddings/vector storage are optional:
    ``RAGPipeline(embedder=None)`` is purely keyword-based and fully offline.
    """

    STAGES = [
        "knowledge_base",
        "embeddings",
        "vector_store",
        "retrieval",
        "context_enrichment",
        "reasoning",
    ]

    def __init__(self, embedder=None, *, chunk_chars: int = DEFAULT_CHUNK_CHARS,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.embedder = embedder
        self.retriever = Retriever()
        self.documents = self.retriever.knowledge_base.get_all_documents()
        self.chunks: list[dict[str, Any]] = []
        self.vector_store = InMemoryVectorStore() if embedder is not None else None
        if self.vector_store is not None:
            self.chunks = chunk_documents(
                self.documents, max_chars=chunk_chars, overlap=chunk_overlap
            )
            for chunk in self.chunks:
                self.vector_store.add(chunk["chunk_id"], self.embedder.embed(chunk["text"]))
            flush = getattr(self.embedder, "flush", None)
            if callable(flush):
                flush()

    @property
    def embedding_stats(self) -> dict[str, Any] | None:
        stats = getattr(self.embedder, "stats", None)
        return dict(stats) if isinstance(stats, dict) else None

    def _documents_from_results(self, results: list[dict]) -> list[dict[str, Any]]:
        """Merge scores into copies of the source documents (never mutate the KB)."""
        documents: list[dict[str, Any]] = []
        for result in results:
            document = dict(result["document"])
            document["score"] = int(result["score"])
            document["source"] = self._source_of(document["id"])
            documents.append(document)
        return documents

    def _source_of(self, document_id: str) -> str:
        return "rules" if str(document_id).startswith("R") else "constraints"

    def retrieval(self, query: str, top_k: int | None = None) -> list[dict]:
        results = self.retriever.retrieve(query, top_k=top_k)
        if self.vector_store is None or self.embedder is None:
            return results
        query_vector = self.embedder.embed(query)
        matches = self.vector_store.search(query_vector, top_k=None)
        keyword_ids = {result["document"]["id"] for result in results}
        best_by_document: dict[str, float] = {}
        for chunk in self.chunks:
            similarity = matches.get(chunk["chunk_id"])
            if similarity is None or chunk["document_id"] not in keyword_ids:
                continue
            document_id = chunk["document_id"]
            best_by_document[document_id] = max(
                best_by_document.get(document_id, float("-inf")), float(similarity)
            )
        if not best_by_document:
            return results
        merged = {result["document"]["id"]: dict(result) for result in results}
        for document_id, similarity in best_by_document.items():
            if document_id in merged:
                merged[document_id]["score"] = max(
                    float(merged[document_id]["score"]), similarity
                )
        return sorted(
            merged.values(),
            key=lambda item: (-item["score"], item["document"]["id"]),
        )

    def run(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        retrieved = self.retrieval(query, top_k=top_k)
        documents = self._documents_from_results(retrieved)
        return {
            "retrieved": retrieved,
            "documents": documents,
            "context": [document["description"] for document in documents],
            "reasoning": None,
            "embedding_stats": self.embedding_stats,
            "chunks": len(self.chunks),
        }
