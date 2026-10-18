import pytest

from connaissance.base_connaissance import KnowledgeBase
from connaissance.regles import RULES, load_rules
from rag.embeddings import HashingEmbedder
from rag.rag import RAGPipeline
from rag.retriever import Retriever


@pytest.fixture
def retriever():
    return Retriever()


def ids(results):
    return [result["document"]["id"] for result in results]


def score_of(results, rule_id):
    return next(result["score"] for result in results if result["document"]["id"] == rule_id)


# --- R2: the retriever returns only the useful rules ----------------------

def test_example_query_returns_exactly_urgency_and_delay_rules(retriever):
    results = retriever.retrieve("Patient P4 is urgent and caregiver S2 is delayed.")

    assert ids(results) == ["R001", "R002"]


def test_constraints_are_never_returned(retriever):
    results = retriever.retrieve("time window schedule")

    assert ids(results) == ["R008"]
    assert all(result["document"]["id"].startswith("R") for result in results)


# --- R3: deterministic keyword scoring ------------------------------------

def test_urgent_emergency_priority_scores_three(retriever):
    results = retriever.retrieve("urgent emergency priority")

    assert score_of(results, "R001") == 3


def test_urgent_patient_scores_one(retriever):
    results = retriever.retrieve("urgent patient")

    assert score_of(results, "R001") == 1
    assert ids(results) == ["R001"]


def test_empty_and_unknown_queries_return_nothing(retriever):
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("weather forecast tomorrow") == []


def test_top_k_limits_and_order_is_stable(retriever):
    results = retriever.retrieve("urgent emergency priority delay", top_k=1)

    assert ids(results) == ["R001"]
    assert retriever.retrieve("urgent emergency priority delay") == retriever.retrieve(
        "urgent emergency priority delay"
    )


# --- R1: knowledge base + six stages --------------------------------------

def test_rules_json_has_required_fields():
    for rule in load_rules():
        assert {"id", "category", "keywords", "text"} <= set(rule)
        assert rule["id"].startswith("R")
        assert rule["keywords"]


def test_required_rule_categories_present():
    categories = {rule["category"] for rule in RULES}

    assert {
        "Priority",
        "Caregiver Delay",
        "Cancellation",
        "Vehicle Breakdown",
        "Conflict",
        "Caregiver Exchange",
        "Skill Compatibility",
        "Time Window",
        "Workload",
    } <= categories


def test_knowledge_base_separates_rules_and_constraints():
    knowledge_base = KnowledgeBase()

    assert all(rule["id"].startswith("R") for rule in knowledge_base.get_rules())
    assert all(constraint["id"].startswith("C") for constraint in knowledge_base.get_constraints())
    assert knowledge_base.get_document("R001")["category"] == "Priority"


def test_pipeline_exposes_six_stages_offline():
    pipeline = RAGPipeline(embedder=None)

    assert len(RAGPipeline.STAGES) == 6
    output = pipeline.run("Patient P4 is urgent and caregiver S2 is delayed.")
    assert ids(output["retrieved"]) == ["R001", "R002"]
    assert len(output["context"]) == 2
    assert output["reasoning"] is None  # stage 6 is optional
    assert pipeline.vector_store is None  # embeddings/storage are optional


def test_hybrid_embedding_path_is_optional_and_offline():
    pipeline = RAGPipeline(embedder=HashingEmbedder())

    assert pipeline.vector_store is not None
    results = pipeline.retrieval("Patient P4 is urgent and caregiver S2 is delayed.")
    ranked = ids(results)

    # The hybrid signal may reorder or add documents, but the two keyword
    # matches must still be present and keep their keyword score.
    assert "R001" in ranked and "R002" in ranked
    assert score_of(results, "R001") >= 1
    assert score_of(results, "R002") >= 1
