import cv2
import numpy as np

from drone_traffic.tracking.cmc import ORBCMC


def test_cmc_static_frame(sample_frame):
    cmc = ORBCMC()
    matrix = cmc.compute(sample_frame, sample_frame)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, np.eye(3), atol=0.1)


def test_cmc_with_shift():
    cmc = ORBCMC()
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    frame2 = cv2.warpAffine(frame1, M, (640, 480))
    matrix = cmc.compute(frame1, frame2)
    assert matrix.shape == (3, 3)


def test_cmc_reset():
    cmc = ORBCMC()
    cmc.reset()
    assert cmc._last_matrix is not None
