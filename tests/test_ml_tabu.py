from ml.tabu import nearest_neighbour_order, tabu_search, tour_cost


def _line(n, spacing=1.0):
    return [(i * spacing, 0.0) for i in range(1, n + 1)]


def test_tour_cost_of_simple_line():
    points = [(1.0, 0.0), (2.0, 0.0)]
    # depot(0,0) -> 1 -> 2 -> depot = 1 + 1 + 2 = 4
    assert tour_cost([0, 1], points, (0.0, 0.0)) == 4.0


def test_nearest_neighbour_starts_closest_to_depot():
    points = _line(4)

    assert nearest_neighbour_order(points, (0.0, 0.0)) == [0, 1, 2, 3]


def test_tabu_returns_a_full_permutation():
    points = [(3.0, 1.0), (1.0, 4.0), (7.0, 2.0), (2.0, 9.0), (5.0, 5.0)]
    result = tabu_search(points, depot=(0.0, 0.0), iterations=40, seed=1)

    assert sorted(result["order"]) == list(range(len(points)))
    assert result["cost"] <= result["start_cost"] + 1e-9


def test_tabu_is_deterministic():
    points = [(3.0, 1.0), (1.0, 4.0), (7.0, 2.0), (2.0, 9.0)]
    first = tabu_search(points, iterations=30, seed=5)
    second = tabu_search(points, iterations=30, seed=5)

    assert first == second


def test_tabu_handles_degenerate_sizes():
    assert tabu_search([], iterations=10)["order"] == []
    single = tabu_search([(5.0, 0.0)], depot=(0.0, 0.0))
    assert single["order"] == [0]
    assert single["cost"] == 10.0
