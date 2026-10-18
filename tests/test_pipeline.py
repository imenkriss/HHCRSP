import main as pipeline


def test_full_pipeline_runs_offline_on_seeded_a21():
    result = pipeline.run_hhcop_pipeline(
        "Patient P1 is urgent and caregiver S2 is delayed",
        dataset="A",
        caregivers=2,
        location=1,
        instance_seed=1,
        population_size=10,
        generations=3,
        local_search=False,
    )

    # LLM stage degrades to the validated fallback when Ollama is absent.
    assert result["llm_output"]["reasoning_source"] == "fallback"
    assert result["instance_info"]["name"] == "A_2_1"

    # A decision is always produced from the non-empty archive.
    assert result["decision"] is not None
    assert "justification" in result["decision"]
    assert isinstance(result["decision"]["visit_plan"], list)

    # Every stage is timed (M1) and Pareto quality is measured.
    complexity = result["complexity"]
    for stage in (
        "Dataset", "RAG", "LLM", "Orchestrator + Agents",
        "NSGA-II Optimization", "Decision Engine", "Metrics", "Total Pipeline",
    ):
        assert "execution_time_sec" in complexity[stage], stage
    assert "pareto_quality" in complexity
    assert "delays_and_overtime" in complexity

    assert result["final_decision"]["status"] in ("assigned", "unassigned")


def test_pipeline_is_deterministic_for_same_seed():
    kwargs = dict(
        dataset="A", caregivers=2, location=1, instance_seed=3,
        population_size=10, generations=3, local_search=False,
    )

    first = pipeline.run_hhcop_pipeline("urgent patient", **kwargs)
    second = pipeline.run_hhcop_pipeline("urgent patient", **kwargs)

    assert first["decision"]["objectives"] == second["decision"]["objectives"]


def test_pipeline_supports_mtcmo_and_ideal_point():
    result = pipeline.run_hhcop_pipeline(
        "urgent patient",
        dataset="A", caregivers=2, location=1, instance_seed=1,
        population_size=10, generations=3, algorithm="mtcmo",
        decision_method="ideal_point",
    )

    assert result["optimization_analysis"]["algorithm"] == "MTCMOLite"
    assert result["decision"]["method"] == "ideal_point"
