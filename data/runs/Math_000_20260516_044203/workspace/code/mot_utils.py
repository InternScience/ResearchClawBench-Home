"""
Multi-Object Tracking Utilities
- Kalman Filter for bounding box motion prediction
- IoU computation
- Hungarian algorithm for data association
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes.
    bbox format: [x1, y1, x2, y2]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def bbox_iou_batch(bboxes1, bboxes2):
    """
    Compute IoU matrix between two sets of bounding boxes.
    Returns matrix of shape (len(bboxes1), len(bboxes2))
    """
    n1, n2 = len(bboxes1), len(bboxes2)
    iou_matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            iou_matrix[i, j] = bbox_iou(bboxes1[i], bboxes2[j])
    return iou_matrix


class KalmanFilter:
    """
    Kalman filter for bounding box tracking.
    State: [x, y, w, h, vx, vy, vw, vh]
    Measurement: [x, y, w, h]
    where (x, y) is the center of the bounding box, (w, h) are width and height.
    """
    
    def __init__(self):
        self.dim_state = 8
        self.dim_meas = 4
        
        # State transition matrix
        self.F = np.eye(self.dim_state)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        
        # Measurement matrix
        self.H = np.zeros((self.dim_meas, self.dim_state))
        for i in range(4):
            self.H[i, i] = 1.0
        
        # Process noise covariance
        self.Q = np.eye(self.dim_state)
        self.Q[0:4, 0:4] *= 1.0
        self.Q[4:8, 4:8] *= 0.01
        
        # Measurement noise covariance
        self.R = np.eye(self.dim_meas) * 10.0
        
        # State covariance
        self.P = np.eye(self.dim_state) * 100.0
        
        # State vector
        self.x = None
        
        self._initialized = False
    
    def init(self, bbox):
        """Initialize the filter with a bounding box [x1, y1, x2, y2]."""
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.x = np.array([x_center, y_center, w, h, 0, 0, 0, 0], dtype=np.float64)
        self._initialized = True
    
    def predict(self):
        """Predict the next state."""
        if not self._initialized:
            return None
        
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()
    
    def update(self, bbox):
        """Update the filter with a new measurement [x1, y1, x2, y2]."""
        if not self._initialized:
            self.init(bbox)
            return
        
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([x_center, y_center, w, h], dtype=np.float64)
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H) @ self.P
    
    def get_bbox(self):
        """Get the current bounding box [x1, y1, x2, y2]."""
        if not self._initialized:
            return None
        
        x_center, y_center, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        x1 = x_center - w / 2.0
        y1 = y_center - h / 2.0
        x2 = x_center + w / 2.0
        y2 = y_center + h / 2.0
        return [x1, y1, x2, y2]
    
    def get_state(self):
        """Get the full state vector."""
        return self.x.copy() if self._initialized else None


def hungarian_match(cost_matrix):
    """
    Perform Hungarian algorithm matching on a cost matrix.
    Returns (row_indices, col_indices) of matched pairs.
    """
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return linear_sum_assignment(cost_matrix)


def iou_cost_matrix(track_bboxes, det_bboxes):
    """
    Compute cost matrix (1 - IoU) for Hungarian matching.
    """
    iou_mat = bbox_iou_batch(track_bboxes, det_bboxes)
    return 1.0 - iou_mat
