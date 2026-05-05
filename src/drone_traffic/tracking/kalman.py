from __future__ import annotations

import numpy as np


class KalmanFilter8:
    DIM_X = 8
    DIM_Z = 4

    def __init__(self) -> None:
        self.x = np.zeros((self.DIM_X, 1))

        self.F = np.eye(self.DIM_X, dtype=np.float64)
        dt = 1.0
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt

        self.H = np.zeros((self.DIM_Z, self.DIM_X), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.P = np.eye(self.DIM_X, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

        std_pos = 8.0
        std_vel = 16.0
        self.Q = np.eye(self.DIM_X, dtype=np.float64)
        self.Q[:4, :4] *= std_pos ** 2
        self.Q[4:, 4:] *= std_vel ** 2

        std_meas = 1.0
        self.R = np.eye(self.DIM_Z, dtype=np.float64) * std_meas ** 2

        self._I = np.eye(self.DIM_X, dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P
        return self.x.copy()

    def get_state(self) -> np.ndarray:
        return self.x.copy().flatten()

    def init_from_bbox(self, cxcyah: np.ndarray) -> None:
        if cxcyah.ndim == 1:
            cxcyah = cxcyah.reshape(-1, 1)
        self.x[:4] = cxcyah[:4]
        self.P = np.eye(self.DIM_X, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

    def apply_affine(self, matrix: np.ndarray) -> None:
        cx, cy = self.x[0, 0], self.x[1, 0]
        w, h = self.x[2, 0], self.x[3, 0]
        pt = np.array([cx, cy, 1.0])
        new_pt = matrix @ pt
        self.x[0, 0] = new_pt[0]
        self.x[1, 0] = new_pt[1]
        self.x[2, 0] = w
        self.x[3, 0] = h
