from __future__ import annotations

import numpy as np

from drone_traffic.core.registry import register_cmc


@register_cmc("orb")
class ORBCMC:
    def __init__(self, max_features: int = 500):
        self._max_features = max_features
        self._detector = None
        self._prev_kp = None
        self._prev_desc = None
        self._last_matrix = np.eye(3, dtype=np.float64)

    def _ensure_detector(self) -> None:
        if self._detector is None:
            import cv2

            self._detector = cv2.ORB_create(nfeatures=self._max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compute(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> np.ndarray:
        import cv2

        self._ensure_detector()

        if prev_frame.ndim == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
        if curr_frame.ndim == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

        kp1, desc1 = self._detector.detectAndCompute(prev_gray, None)
        kp2, desc2 = self._detector.detectAndCompute(curr_gray, None)

        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return self._last_matrix

        matches = self._matcher.match(desc1, desc2)
        if len(matches) < 4:
            return self._last_matrix

        matches = sorted(matches, key=lambda m: m.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if matrix is not None:
            result = np.eye(3, dtype=np.float64)
            result[:2, :] = matrix
            self._last_matrix = result
            return result

        return self._last_matrix

    def reset(self) -> None:
        self._prev_kp = None
        self._prev_desc = None
        self._last_matrix = np.eye(3, dtype=np.float64)
