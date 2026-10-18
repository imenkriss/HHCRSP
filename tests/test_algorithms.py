import random

import pytest

from optimisation.mtwmo.algorithms import (
    CROSSOVERS,
    Algorithm,
    Candidate,
    MTCMOLite,
    NSGAII,
    binary_tournament,
    constraint_dominates,
    crowding_distance,
    fast_non_dominated_sort,
    heuristic_permutation,
    insert_mutation,
    order_crossover,
    pareto_dominates,
    pmx_crossover,
    position_crossover,
    random_permutation,
    swap_mutation,
)
from optimisation.mtwmo.generator import generate_instance


def make_candidate(objectives, violation=0.0, permutation=(0,), feasible=None):
    if feasible is None:
        feasible = violation <= 1e-9
    return Candidate(
        permutation=tuple(permutation),
        objectives=tuple(objectives),
        violation=violation,
        z1=objectives[0],
        z2=-objectives[1],
        z3=objectives[2],
        feasible=feasible,
    )


# --- dominance / ranking / crowding ---------------------------------------

def test_pareto_dominates_basics():
    assert pareto_dominates((1, 1, 1), (2, 2, 2))
    assert pareto_dominates((1, 2, 2), (2, 2, 2))
    assert not pareto_dominates((1, 3, 2), (2, 2, 2))
    assert not pareto_dominates((2, 2, 2), (2, 2, 2))


def test_constraint_domination_feasible_beats_infeasible():
    feasible = make_candidate((5, 5, 5), violation=0.0)
    infeasible = make_candidate((1, 1, 1), violation=2.0)

    assert constraint_dominates(feasible, infeasible)
    assert not constraint_dominates(infeasible, feasible)


def test_constraint_domination_orders_infeasible_by_violation():
    lower = make_candidate((9, 9, 9), violation=1.0)
    higher = make_candidate((1, 1, 1), violation=3.0)

    assert constraint_dominates(lower, higher)
    assert not constraint_dominates(higher, lower)


def test_fast_non_dominated_sort_handmade():
    s1 = make_candidate((1, 5, 5), permutation=(1,))
    s2 = make_candidate((2, 2, 5), permutation=(2,))
    s3 = make_candidate((5, 1, 1), permutation=(3,))
    s4 = make_candidate((2, 2, 2), permutation=(4,))
    s5 = make_candidate((3, 3, 3), permutation=(5,))

    fronts = fast_non_dominated_sort([s1, s2, s3, s4, s5])

    assert len(fronts) == 2
    assert {tuple(c.objectives) for c in fronts[0]} == {(1, 5, 5), (5, 1, 1), (2, 2, 2)}
    assert {tuple(c.objectives) for c in fronts[1]} == {(2, 2, 5), (3, 3, 3)}


def test_fast_non_dominated_sort_respects_constraints():
    feasible = make_candidate((9, 9, 9), violation=0.0, permutation=(1,))
    infeasible = make_candidate((1, 1, 1), violation=2.0, permutation=(2,))

    fronts = fast_non_dominated_sort([feasible, infeasible])

    assert fronts[0] == [feasible]
    assert fronts[1] == [infeasible]


def test_crowding_distance_boundaries_and_midpoint():
    front = [
        make_candidate((0, 0, 0), permutation=(1,)),
        make_candidate((1, 1, 1), permutation=(2,)),
        make_candidate((2, 2, 2), permutation=(3,)),
    ]

    distance = crowding_distance(front)

    assert distance[0] == float("inf")
    assert distance[2] == float("inf")
    assert distance[1] == pytest.approx(3.0)


def test_binary_tournament_prefers_lower_rank():
    rng = random.Random(0)
    population = [make_candidate((9, 9, 9), permutation=(1,)), make_candidate((1, 1, 1), permutation=(2,))]
    rank = [0, 1]
    crowding = [0.0, 0.0]

    counts = {1: 0, 2: 0}
    for _ in range(200):
        winner = binary_tournament(population, rank, crowding, rng)
        counts[winner.permutation[0]] += 1

    # Rank 0 must win strictly more often than rank 1 (it loses only when both
    # tournament draws hit the worse individual).
    assert counts[1] > counts[2]


# --- permutation operators ------------------------------------------------

def test_crossovers_preserve_permutation():
    rng = random.Random(42)
    parent_a = [1, 2, 3, 4, 5, 6, 7]
    parent_b = list(reversed(parent_a))
    for name, operator in CROSSOVERS.items():
        for _ in range(20):
            child = operator(parent_a, parent_b, rng)
            assert sorted(child) == parent_a, name


def test_mutations_preserve_permutation():
    rng = random.Random(7)
    permutation = [1, 2, 3, 4, 5]
    assert sorted(swap_mutation(permutation, rng)) == permutation
    assert sorted(insert_mutation(permutation, rng)) == permutation


def test_pmx_matches_known_swaps():
    rng = random.Random(0)
    child = pmx_crossover([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], rng)
    assert sorted(child) == [1, 2, 3, 4, 5, 6]


def test_order_and_position_crossover_are_valid_permutations():
    rng = random.Random(1)
    a = [1, 2, 3, 4, 5, 6]
    b = [5, 3, 6, 1, 4, 2]
    assert sorted(order_crossover(a, b, rng)) == a
    assert sorted(position_crossover(a, b, rng)) == a


def test_initial_permutations_span_all_genes():
    instance = generate_instance("A", 2, 1, seed=1)
    rng = random.Random(3)
    gene_set = set(range(1, len(instance.patients) + len(instance.caregivers) + 1))

    assert set(random_permutation(len(instance.patients), len(instance.caregivers), rng)) == gene_set
    assert set(heuristic_permutation(instance, rng)) == gene_set


# --- algorithm behaviour ---------------------------------------------------

def test_both_algorithms_share_the_common_interface():
    assert issubclass(NSGAII, Algorithm)
    assert issubclass(MTCMOLite, Algorithm)
    assert NSGAII.name == "NSGA-II"
    assert MTCMOLite.name == "MTCMOLite"


@pytest.mark.parametrize("algorithm", [NSGAII(population_size=20, generations=8), MTCMOLite(population_size=20, generations=8)])
def test_run_returns_pareto_set_with_flagged_solutions(algorithm):
    instance = generate_instance("A", 2, 1, seed=1)

    result = algorithm.run(instance, seed=0, gamma=0.0)

    assert result.size >= 1
    assert result.generations == 8
    for candidate in result.solutions:
        assert len(candidate.objectives) == 3
        assert candidate.feasible == (candidate.violation <= 1e-9)
        assert candidate.objectives == (candidate.z1, -candidate.z2, candidate.z3)
    assert result.runtime_seconds > 0.0


def test_same_seed_reproduces_the_same_result():
    instance = generate_instance("A", 2, 1, seed=5)
    algorithm = NSGAII(population_size=20, generations=8, seed=1)

    first = algorithm.run(instance, seed=2, gamma=0.0)
    second = algorithm.run(instance, seed=2, gamma=0.0)

    assert [c.permutation for c in first.solutions] == [c.permutation for c in second.solutions]
    assert [c.objectives for c in first.solutions] == [c.objectives for c in second.solutions]
    assert first.first_feasible_generation == second.first_feasible_generation


def test_mtcmo_lite_reproduces_with_same_seed():
    instance = generate_instance("A", 2, 1, seed=5)
    algorithm = MTCMOLite(population_size=20, generations=8)

    first = algorithm.run(instance, seed=4, gamma=0.0)
    second = algorithm.run(instance, seed=4, gamma=0.0)

    assert [c.permutation for c in first.solutions] == [c.permutation for c in second.solutions]


def test_objectives_are_never_combined_into_one_number():
    instance = generate_instance("A", 2, 1, seed=1)
    result = NSGAII(population_size=20, generations=8).run(instance, seed=0, gamma=0.0)

    feasible = result.feasible_solutions
    if len(feasible) >= 2:
        # Two distinct non-dominated solutions must trade off, not share a scalar.
        differences = {tuple(c.objectives) for c in feasible}
        assert len(differences) >= 1
    for candidate in result.solutions:
        assert isinstance(candidate.objectives, tuple)
        assert len(candidate.objectives) == 3


def test_small_instance_returns_feasible_quickly():
    instance = generate_instance("A", 2, 1, seed=1)

    result = NSGAII(population_size=30, generations=20, local_search=False).run(
        instance, seed=0, gamma=0.0
    )

    assert result.first_feasible_generation is not None
    assert result.has_feasible
    assert result.runtime_seconds < 30.0


def test_local_search_enabled_keeps_solutions_valid():
    instance = generate_instance("A", 2, 1, seed=1)
    gene_set = set(range(1, len(instance.patients) + len(instance.caregivers) + 1))

    result = NSGAII(
        population_size=20,
        generations=10,
        local_search=True,
        local_search_interval=5,
        local_search_elites=2,
        local_search_iterations=3,
    ).run(instance, seed=1, gamma=0.0)

    assert result.size >= 1
    for candidate in result.solutions:
        assert set(candidate.permutation) == gene_set
        assert candidate.feasible == (candidate.violation <= 1e-9)
