import pytest

from ml.pipeline import (
    ClusterRoute,
    RoutingResult,
    compare_svm_vs_baseline,
    run_pipeline,
)


@pytest.fixture
def synthetic():
    points = [(float(i % 6) * 5.0, float(i // 6) * 5.0) for i in range(30)]
    care = [i % 2 for i in range(30)]
    return points, care


def test_run_pipeline_structure(synthetic):
    points, care = synthetic
    result = run_pipeline(points, (0.0, 0.0), care, k=3, seed=4,
                          tabu_kwargs={"iterations": 20})

    assert isinstance(result, RoutingResult)
    assert result.method == "svm+kmeans+tabu"
    assert result.use_care_type is True
    assert result.n == 30
    assert result.k == 3
    assert all(isinstance(route, ClusterRoute) for route in result.clusters)
    # Every customer is routed exactly once across the clusters.
    routed = sorted(node for route in result.clusters for node in route.members)
    assert routed == list(range(30))
    assert result.total_cost == pytest.approx(
        sum(route.cost for route in result.clusters), rel=1e-9
    )


def test_baseline_ignores_care_type(synthetic):
    points, care = synthetic
    baseline = run_pipeline(points, (0.0, 0.0), care, k=3, seed=4,
                            use_svm=False, tabu_kwargs={"iterations": 20})

    assert baseline.method == "kmeans+tabu"
    assert baseline.use_care_type is False


def test_compare_reports_measured_fields(synthetic):
    points, care = synthetic
    comparison = compare_svm_vs_baseline(
        points, (0.0, 0.0), care, k=3, seed=4, tabu_kwargs={"iterations": 20}
    )

    assert set(comparison) >= {
        "svm", "baseline", "svm_total_cost", "baseline_total_cost",
        "improvement_percent",
    }
    expected = (
        comparison["baseline_total_cost"] - comparison["svm_total_cost"]
    ) / comparison["baseline_total_cost"] * 100.0
    assert comparison["improvement_percent"] == pytest.approx(expected, rel=1e-9)
    assert comparison["svm"].method == "svm+kmeans+tabu"
    assert comparison["baseline"].method == "kmeans+tabu"
