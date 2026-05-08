import numpy as np
from drone_traffic.tracking.matching import (
    combined_cost_matrix,
    cosine_distance,
    hungarian_assignment,
    iou_distance,
)


def test_iou_distance_zero():
    tracks = np.array([[100, 100, 50, 50]])
    dets = np.array([[100, 100, 50, 50]])
    dist = iou_distance(tracks, dets)
    assert dist[0, 0] < 0.01


def test_iou_distance_far():
    tracks = np.array([[100, 100, 50, 50]])
    dets = np.array([[500, 500, 50, 50]])
    dist = iou_distance(tracks, dets)
    assert dist[0, 0] > 0.9


def test_hungarian_assignment():
    cost = np.array([[0.1, 0.8], [0.7, 0.2]])
    matches, un_r, un_c = hungarian_assignment(cost, threshold=0.5)
    assert len(matches) == 2
    assert (0, 0) in matches
    assert (1, 1) in matches


def test_hungarian_empty():
    matches, un_r, un_c = hungarian_assignment(np.zeros((0, 0)), threshold=1.0)
    assert matches == []


def test_cosine_distance_identical():
    features = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dist = cosine_distance(features, features)
    np.testing.assert_almost_equal(dist[0, 0], 0.0, decimal=5)
    np.testing.assert_almost_equal(dist[1, 1], 0.0, decimal=5)


def test_cosine_distance_orthogonal():
    t = np.array([[1.0, 0.0]])
    d = np.array([[0.0, 1.0]])
    dist = cosine_distance(t, d)
    np.testing.assert_almost_equal(dist[0, 0], 1.0, decimal=5)


def test_cosine_distance_empty():
    dist = cosine_distance(np.zeros((0, 3)), np.array([[1.0, 0.0, 0.0]]))
    assert dist.shape == (0, 1)


def test_cosine_distance_normalized():
    t = np.array([[3.0, 4.0]])
    d = np.array([[6.0, 8.0]])
    dist = cosine_distance(t, d)
    np.testing.assert_almost_equal(dist[0, 0], 0.0, decimal=5)


def test_combined_cost_matrix():
    iou_cost = np.array([[0.1, 0.8], [0.7, 0.2]])
    app_cost = np.array([[0.05, 0.9], [0.8, 0.1]])

    combined = combined_cost_matrix(iou_cost, app_cost)

    assert combined.shape == iou_cost.shape
    assert combined[0, 0] < 0.5
    assert combined[0, 1] == 1.0
    assert combined[1, 0] == 1.0


def test_combined_cost_gating():
    iou_cost = np.array([[0.95]])
    app_cost = np.array([[0.05]])

    combined = combined_cost_matrix(
        iou_cost, app_cost, proximity_thresh=0.5
    )
    assert combined[0, 0] == 1.0


def test_combined_cost_appearance_gating():
    iou_cost = np.array([[0.1]])
    app_cost = np.array([[0.9]])

    combined = combined_cost_matrix(
        iou_cost, app_cost, appearance_thresh=0.25
    )
    assert combined[0, 0] == 1.0


def test_combined_cost_shape_mismatch():
    iou_cost = np.array([[0.1, 0.2]])
    app_cost = np.array([[0.1, 0.2, 0.3]])

    combined = combined_cost_matrix(iou_cost, app_cost)
    assert np.array_equal(combined, iou_cost)
