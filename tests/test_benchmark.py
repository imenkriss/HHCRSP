import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_benchmark import (  # noqa: E402
    NA,
    aggregate,
    build_instance_list,
    run_one,
)
from optimisation.mtwmo.algorithms import Candidate, ParetoSet  # noqa: E402


def _pareto_set(solutions, first_feasible=0):
    return ParetoSet(
        algorithm="NSGA-II",
        instance_name="A_2_1",
        seed=0,
        gamma=0.0,
        solutions=solutions,
        first_feasible_generation=first_feasible,
        generations=1,
        runtime_seconds=0.1,
    )


def test_build_instance_list_defaults_to_a_and_b_caregiver_counts():
    names = [row["name"] for row in build_instance_list(("A",))]

    assert names == ["A_2_1", "A_3_1", "A_4_1"]


def test_aggregate_marks_instances_without_feasible_solution():
    infeasible = Candidate(
        permutation=(1,),
        objectives=(1.0, -2.0, 3.0),
        violation=5.0,
        z1=1.0,
        z2=2.0,
        z3=3.0,
        feasible=False,
    )

    row = aggregate("A_2_1", "A", "NSGA-II", 0.0, [_pareto_set([infeasible])])

    assert row["no_feasible_solution"] is True
    assert row["best_z1"] == NA
    assert row["feasible_runs"] == 0


def test_aggregate_reports_best_and_mean_over_feasible_solutions():
    feasible = Candidate(
        permutation=(1,),
        objectives=(4.0, -6.0, 2.0),
        violation=0.0,
        z1=4.0,
        z2=6.0,
        z3=2.0,
        feasible=True,
        caregivers_used=2,
    )
    better = Candidate(
        permutation=(2,),
        objectives=(2.0, -5.0, 1.0),
        violation=0.0,
        z1=2.0,
        z2=5.0,
        z3=1.0,
        feasible=True,
        caregivers_used=1,
    )

    row = aggregate("A_2_1", "A", "NSGA-II", 0.0, [_pareto_set([feasible, better])])

    assert row["best_z1"] == 2.0
    assert row["best_z2"] == 6.0
    assert row["best_z3"] == 1.0
    assert row["non_dominated_solutions"] == 2
    assert row["mean_pareto_solutions"] == 2


def test_run_one_is_reproducible():
    info = {"dataset": "A", "caregivers": 2, "location": 1}

    first = run_one(info, "NSGA-II", 0, 0.0, 10, 3)["result"]
    second = run_one(info, "NSGA-II", 0, 0.0, 10, 3)["result"]

    assert [c.permutation for c in first.solutions] == [c.permutation for c in second.solutions]
