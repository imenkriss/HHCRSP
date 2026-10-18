from llm.raisonner import LLMRaisonner, ReasoningResult


def test_analyze_falls_back_without_ollama():
    reasoner = LLMRaisonner(model="does-not-matter")

    result = reasoner.analyze("Patient P4 is urgent and caregiver S2 is delayed.", [])

    assert result["reasoning_source"] == "fallback"
    # The fallback must still match the validated schema.
    ReasoningResult.model_validate({key: result[key] for key in ReasoningResult.model_fields})


def test_fallback_classifies_urgent_request():
    result = LLMRaisonner().analyze("Patient P4 is emergency priority", [])

    assert result["request_type"] == "patient_urgent"
    assert result["patient_id"] == "P4"
    assert result["urgency"] is True


def test_fallback_classifies_delayed_caregiver():
    result = LLMRaisonner().analyze("caregiver S2 is delayed by traffic", [])

    assert result["request_type"] == "caregiver_info"
    assert result["caregiver_id"] == "S2"
    assert result["delayed"] is True


def test_fallback_unknown_query_invents_nothing():
    result = LLMRaisonner().analyze("weather forecast tomorrow", [])

    assert result["request_type"] == "unknown"
    assert result["patient_id"] is None
    assert result["caregiver_id"] is None
    assert result["urgency"] is False
    assert result["delayed"] is False


def test_fallback_records_rule_ids_from_retrieved_documents():
    retrieved = [
        {"id": "R001", "text": "urgent rule"},
        {"id": "R002", "text": "delay rule"},
        {"id": "C003", "text": "time window constraint"},
    ]

    result = LLMRaisonner().analyze("urgent patient", retrieved)

    assert result["relevant_rules"] == ["R001", "R002"]
    assert result["constraints"] == ["C003"]
