import numpy as np

from ml.clustering import build_feature_matrix, cluster_points, kmeans_clusters


def _grid(n, scale=10.0):
    return [(float(i % 5) * scale, float(i // 5) * scale) for i in range(n)]


def test_feature_matrix_includes_care_type_when_requested():
    points = [(1.0, 2.0), (3.0, 4.0)]
    with_care = build_feature_matrix(points, [0, 1], use_care_type=True)
    without = build_feature_matrix(points, None, use_care_type=False)

    assert with_care.shape == (2, 3)
    assert without.shape == (2, 2)
    assert with_care[:, 0].tolist() == [0.0, 1.0]


def test_cluster_points_partitions_all_indices():
    points = _grid(30)
    clustering = cluster_points(points, [i % 2 for i in range(30)], k=3, seed=7)

    flattened = sorted(index for members in clustering.clusters.values() for index in members)
    assert flattened == list(range(30))
    assert clustering.k == 3
    assert all(size > 0 for size in clustering.sizes().values())


def test_kmeans_is_deterministic():
    points = np.array(_grid(20))
    first = kmeans_clusters(points, k=3, seed=3)
    second = kmeans_clusters(points, k=3, seed=3)

    assert np.array_equal(first, second)


def test_k_is_capped_at_number_of_points():
    points = [(0.0, 0.0), (1.0, 1.0)]
    clustering = cluster_points(points, [0, 1], k=5, seed=1)

    assert clustering.k == 2


def test_care_type_changes_clustering():
    # Two spatial groups but a care type that cuts across them.
    points = [(0.0, 0.0), (0.1, 0.0), (10.0, 0.0), (10.1, 0.0)]
    care = [0, 1, 0, 1]
    with_care = cluster_points(points, care, k=2, seed=0, use_care_type=True)
    without = cluster_points(points, care, k=2, seed=0, use_care_type=False)

    assert with_care.labels.tolist() != without.labels.tolist()
