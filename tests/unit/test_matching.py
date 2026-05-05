import numpy as np
from drone_traffic.tracking.matching import iou_distance, hungarian_assignment


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
