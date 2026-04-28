"""
tracker_core.py
================
Shared utilities for ByteTrack and SparseTrack:
- IoU computation
- Constant-velocity Kalman filter for [cx, cy, a=w/h, h, vx, vy, va, vh]
  (the standard SORT/ByteTrack state)
- STrack class
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional


# ---------- bbox utilities ----------

def xyxy_to_xyah(b):
    x1, y1, x2, y2 = b
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w / h, h], dtype=np.float64)


def xyah_to_xyxy(s):
    cx, cy, a, h = s
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


def iou_matrix(boxes_a, boxes_b):
    """boxes_a: (N,4) xyxy, boxes_b: (M,4) xyxy -> (N,M) IoU matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    A = np.asarray(boxes_a, dtype=np.float64)
    B = np.asarray(boxes_b, dtype=np.float64)
    a_area = (A[:, 2] - A[:, 0]).clip(min=0) * (A[:, 3] - A[:, 1]).clip(min=0)
    b_area = (B[:, 2] - B[:, 0]).clip(min=0) * (B[:, 3] - B[:, 1]).clip(min=0)
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a_area[:, None] + b_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------- Kalman filter ----------

class KalmanFilterXYAH:
    """
    Constant-velocity Kalman filter on the 8-D state
        x = [cx, cy, a, h, vx, vy, va, vh]
    (SORT/ByteTrack convention).
    """

    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        ndim, dt = 4, 1.0
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, ndim + i] = dt
        self._H = np.eye(4, 8)
        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity

    def initiate(self, measurement: np.ndarray):
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        h = max(measurement[3], 1.0)
        std = [
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * h,
            1e-2,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * h,
            1e-5,
            10 * self.std_weight_velocity * h,
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        h = max(mean[3], 1.0)
        std_pos = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ]
        std_vel = [
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        h = max(mean[3], 1.0)
        std = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        innov = measurement - self._H @ mean
        mean = mean + K @ innov
        cov = (np.eye(8) - K @ self._H) @ cov
        return mean, cov


# ---------- STrack ----------

class STrack:
    _global_count = 0
    shared_kf: KalmanFilterXYAH = KalmanFilterXYAH()

    def __init__(self, bbox_xyxy: np.ndarray, score: float, frame_id: int):
        self.score = float(score)
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.tracklet_len = 1
        self.time_since_update = 0
        self.state = "tentative"
        STrack._global_count += 1
        self.track_id = STrack._global_count
        self.history: list = []  # list of (frame_id, xyxy)
        meas = xyxy_to_xyah(bbox_xyxy)
        self.mean, self.cov = STrack.shared_kf.initiate(meas)
        self.history.append((frame_id, bbox_xyxy.copy()))

    @classmethod
    def reset_id(cls):
        cls._global_count = 0

    def predict(self):
        self.mean, self.cov = STrack.shared_kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray, score: float, frame_id: int):
        meas = xyxy_to_xyah(bbox_xyxy)
        self.mean, self.cov = STrack.shared_kf.update(self.mean, self.cov, meas)
        self.score = float(score)
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.tracklet_len >= 2:
            self.state = "tracked"
        else:
            self.state = "tracked"
        self.history.append((frame_id, bbox_xyxy.copy()))

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    @property
    def bottom_y(self) -> float:
        return float(self.bbox[3])
