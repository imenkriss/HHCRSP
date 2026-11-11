"""Test F — time and peak-memory measurements on a representative workload.

Bounds are deliberately loose: they detect a pathological regression (an
accidental 1000-patient run, an unbounded loop) without being flaky on a slow
Intel i5 laptop. The measured numbers are reported through the recorded
``ResourceMonitor`` payload and through pytest's ``record_property`` hook.
"""

from __future__ import annotations

import math

import pytest

import main
from complexity.metrics import ResourceMonitor
from data import dataset_loader


def test_loading_a_representative_working_set_is_bounded(record_property):
    monitor = ResourceMonitor().start()
    patients = dataset_loader.load_hhcop_patients(limit=200, area=100.0)
    usage = monitor.stop()

    record_property("load_wall_sec", usage["wall_time_sec"])
    record_property("load_peak_python_mb", usage["peak_python_mb"])
    record_property("load_peak_rss_mb", usage["peak_rss_mb"])

    assert len(patients) == 200
    assert len({patient["id"] for patient in patients}) == 200
    assert usage["wall_time_sec"] < 30.0
    assert usage["peak_python_mb"] >= 0.0
    assert usage["peak_rss_mb"] > 0.0
    assert usage["cpu_count"] >= 1
    assert math.isfinite(usage["wall_time_sec"])


@pytest.mark.slow
def test_pipeline_reports_time_and_memory(record_property):
    result = main.run_hhcop_pipeline(
        "Patient P1 is urgent",
        dataset="A",
        caregivers=2,
        location=1,
        instance_seed=1,
        population_size=10,
        generations=3,
        local_search=False,
        compare_algorithms=True,
        comparison_runs=1,
        comparison_population=8,
        comparison_generations=3,
    )
    resources = result["complexity"]["resources"]
    record_property("pipeline_wall_sec", resources["wall_time_sec"])
    record_property("pipeline_peak_python_mb", resources["peak_python_mb"])
    record_property("pipeline_peak_rss_mb", resources["peak_rss_mb"])

    assert resources["wall_time_sec"] < 120.0
    assert resources["peak_python_mb"] > 0.0
    assert resources["peak_rss_mb"] > 0.0
    assert result["complexity"]["Total Pipeline"]["execution_time_sec"] >= 0.0
    for stage in ("Dataset", "RAG", "LLM", "NSGA-II Optimization", "Decision Engine"):
        assert "execution_time_sec" in result["complexity"][stage]
    # The live comparison measured both algorithms on the same instance.
    assert result["comparison"]["algorithm_status"]["available"] is True
    assert len(result["comparison"]["algorithms"]) == 2
