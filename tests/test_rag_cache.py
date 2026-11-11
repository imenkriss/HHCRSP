"""Test D — RAG ingestion, chunking, embedding persistence and retrieval metadata."""

from __future__ import annotations

import json

from connaissance.base_connaissance import KnowledgeBase
from rag.embeddings import CachedEmbedder, HashingEmbedder
from rag.rag import RAGPipeline, chunk_document, chunk_documents


def _pipeline(tmp_path, dimensions=64):
    embedder = CachedEmbedder(HashingEmbedder(dimensions=dimensions), cache_dir=tmp_path)
    return RAGPipeline(embedder=embedder), embedder


def test_chunking_preserves_document_metadata():
    document = {
        "id": "R999",
        "category": "Priority",
        "text": "a" * 250,
        "description": "long rule",
    }
    chunks = chunk_document(document, max_chars=100, overlap=20)

    assert len(chunks) == 3
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    for index, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"R999#c{index}"
        assert chunk["document_id"] == "R999"
        assert chunk["document"] is document
        assert 0 < len(chunk["text"]) <= 100
    # Overlap really overlaps: chunk 1 starts before chunk 0 ends.
    assert chunks[0]["text"][-20:] == chunks[1]["text"][:20]


def test_chunking_empty_document_yields_nothing():
    assert chunk_document({"id": "R1", "text": "   "}) == []
    assert chunk_documents([]) == []


def test_every_knowledge_document_is_ingested_once(tmp_path):
    knowledge_base = KnowledgeBase()
    pipeline, _ = _pipeline(tmp_path)

    document_ids = {document["id"] for document in knowledge_base.get_all_documents()}
    chunk_document_ids = {chunk["document_id"] for chunk in pipeline.chunks}
    assert chunk_document_ids == document_ids
    assert len(pipeline.vector_store) == len(pipeline.chunks)
    assert len({chunk["chunk_id"] for chunk in pipeline.chunks}) == len(pipeline.chunks)


def test_retrieval_returns_expected_ids_and_scores(tmp_path):
    pipeline, _ = _pipeline(tmp_path)

    outcome = pipeline.run("Patient P4 is urgent and caregiver S2 is delayed.")
    identifiers = [document["id"] for document in outcome["documents"]]

    assert identifiers == ["R001", "R002"]
    assert all(document["source"] == "rules" for document in outcome["documents"])
    assert all(document["score"] >= 1 for document in outcome["documents"])
    assert outcome["context"] == [document["description"] for document in outcome["documents"]]
    assert outcome["reasoning"] is None


def test_embeddings_are_persisted_and_reused(tmp_path):
    pipeline, embedder = _pipeline(tmp_path)
    cache_path = embedder.cache.path
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(payload["vectors"]) == len(pipeline.chunks)
    # Vectors are L2-normalised (cosine == dot product in the store).
    for vector in payload["vectors"].values():
        norm = sum(value * value for value in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    second = CachedEmbedder(HashingEmbedder(dimensions=64), cache_dir=tmp_path)
    text = pipeline.chunks[0]["text"]
    assert second.embed(text) == embedder.embed(text)
    assert second.hits == 1
    assert second.misses == 0


def test_rebuilding_the_pipeline_does_not_duplicate_vectors(tmp_path):
    first, _ = _pipeline(tmp_path)
    second, _ = _pipeline(tmp_path)

    assert len(second.vector_store) == len(first.vector_store)
    assert [chunk["chunk_id"] for chunk in second.chunks] == [
        chunk["chunk_id"] for chunk in first.chunks
    ]
    assert second.embedding_stats["hits"] == len(second.chunks)
    assert second.embedding_stats["misses"] == 0


def test_unknown_query_returns_no_documents(tmp_path):
    pipeline, _ = _pipeline(tmp_path)

    outcome = pipeline.run("weather forecast tomorrow")

    assert outcome["documents"] == []
    assert outcome["context"] == []
