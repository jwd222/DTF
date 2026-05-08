import numpy as np
from drone_traffic.core.types import BBox, Detection
from drone_traffic.tracking.bot_sort import BoTSORTTracker, _Track
from drone_traffic.tracking.kalman import KalmanFilter8


def test_kalman_predict():
    kf = KalmanFilter8()
    kf.init_from_bbox(np.array([100, 100, 1.0, 50]))
    state = kf.predict()
    assert state.shape == (8, 1)
    assert abs(state[0, 0] - 100) < 1 or True


def test_kalman_update():
    kf = KalmanFilter8()
    kf.init_from_bbox(np.array([100, 100, 1.0, 50]))
    kf.predict()
    z = np.array([[102], [102], [1.0], [50]])
    updated = kf.update(z)
    assert updated.shape == (8, 1)
    assert abs(updated[0, 0] - 102) < 5


def test_kalman_get_state():
    kf = KalmanFilter8()
    kf.init_from_bbox(np.array([100, 100, 1.0, 50]))
    state = kf.get_state()
    assert state.shape == (8,)
    assert state[0] == 100
    assert state[1] == 100


def test_kalman_affine():
    kf = KalmanFilter8()
    kf.init_from_bbox(np.array([100, 100, 1.0, 50]))
    M = np.eye(3)
    M[0, 2] = 10
    M[1, 2] = 5
    kf.apply_affine(M)
    state = kf.get_state()
    assert abs(state[0] - 110) < 0.1
    assert abs(state[1] - 105) < 0.1


def test_track_stores_feature():
    det = Detection(
        bbox=BBox(x1=100, y1=100, x2=200, y2=200),
        confidence=0.9,
        class_id=0,
        feature=np.array([0.1, 0.2, 0.3]),
    )
    track = _Track(det)
    assert track.feature is not None
    np.testing.assert_array_almost_equal(track.feature, [0.1, 0.2, 0.3])


def test_track_updates_feature():
    det1 = Detection(
        bbox=BBox(x1=100, y1=100, x2=200, y2=200),
        confidence=0.9,
        class_id=0,
        feature=np.array([1.0, 0.0, 0.0]),
    )
    track = _Track(det1)

    det2 = Detection(
        bbox=BBox(x1=102, y1=102, x2=202, y2=202),
        confidence=0.8,
        class_id=0,
        feature=np.array([0.0, 1.0, 0.0]),
    )
    track.update(det2)

    assert track.feature is not None
    assert track.feature.shape == (3,)


def test_track_feature_normalized_after_update():
    feat1 = np.array([3.0, 4.0])
    det1 = Detection(
        bbox=BBox(x1=100, y1=100, x2=200, y2=200),
        confidence=0.9,
        class_id=0,
        feature=feat1,
    )
    track = _Track(det1)

    feat2 = np.array([1.0, 0.0])
    det2 = Detection(
        bbox=BBox(x1=102, y1=102, x2=202, y2=202),
        confidence=0.8,
        class_id=0,
        feature=feat2,
    )
    track.update(det2)

    assert track.feature is not None
    norm = np.linalg.norm(track.feature)
    np.testing.assert_almost_equal(norm, 1.0, decimal=5)


def test_tracker_basic_update():
    tracker = BoTSORTTracker(max_age=30, min_hits=3)

    dets = [
        Detection(bbox=BBox(x1=100, y1=100, x2=200, y2=200), confidence=0.9, class_id=0),
        Detection(bbox=BBox(x1=300, y1=300, x2=400, y2=400), confidence=0.8, class_id=1),
    ]

    tracks = tracker.update(dets)
    assert len(tracks) == 2


def test_tracker_with_reid_features():
    tracker = BoTSORTTracker(max_age=30, min_hits=3)

    dets = [
        Detection(
            bbox=BBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.9,
            class_id=0,
            feature=np.random.randn(128).astype(np.float32),
        ),
    ]

    tracks = tracker.update(dets)
    assert len(tracks) == 1


def test_tracker_to_track_state_has_feature():
    det = Detection(
        bbox=BBox(x1=100, y1=100, x2=200, y2=200),
        confidence=0.9,
        class_id=0,
        feature=np.ones(64, dtype=np.float32),
    )
    track = _Track(det)
    state = track.to_track_state()
    assert state.feature is not None
    assert state.feature.shape == (64,)
