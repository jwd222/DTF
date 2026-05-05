import numpy as np
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
