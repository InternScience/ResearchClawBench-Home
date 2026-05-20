"""
Kalman Filter implementation for multi-object tracking.
State: [x_c, y_c, w, h, dx, dy, dw, dh]
"""
import numpy as np

class KalmanFilter:
    def __init__(self):
        # State dimension: 8 (x_c, y_c, w, h, dx, dy, dw, dh)
        self.dim_x = 8
        # Measurement dimension: 4 (x_c, y_c, w, h)
        self.dim_z = 4
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # x += dx
        self.F[1, 5] = 1  # y += dy
        self.F[2, 6] = 1  # w += dw
        self.F[3, 7] = 1  # h += dh
        
        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        
    def initiate(self, measurement):
        """Initialize state from first measurement [x1, y1, x2, y2]"""
        x_c = (measurement[0] + measurement[2]) / 2.0
        y_c = (measurement[1] + measurement[3]) / 2.0
        w = measurement[2] - measurement[0]
        h = measurement[3] - measurement[1]
        
        self.x = np.array([x_c, y_c, w, h, 0, 0, 0, 0])
        
        # Initialize covariance
        self.P = np.eye(self.dim_x)
        self.P[0, 0] *= 10  # x_c uncertainty
        self.P[1, 1] *= 10  # y_c uncertainty
        self.P[2, 2] *= 10  # w uncertainty
        self.P[3, 3] *= 10  # h uncertainty
        self.P[4, 4] *= 100  # dx uncertainty
        self.P[5, 5] *= 100  # dy uncertainty
        self.P[6, 6] *= 100  # dw uncertainty
        self.P[7, 7] *= 100  # dh uncertainty
        
    def predict(self):
        """Predict state to next time step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T
        
        # Add process noise
        Q = np.eye(self.dim_x)
        Q[0, 0] = 1.0   # position noise
        Q[1, 1] = 1.0
        Q[2, 2] = 0.1   # size noise
        Q[3, 3] = 0.1
        Q[4, 4] = 0.1   # velocity noise
        Q[5, 5] = 0.1
        Q[6, 6] = 0.01
        Q[7, 7] = 0.01
        self.P += Q
        
        return self.get_state()
    
    def update(self, measurement):
        """Update state with measurement [x1, y1, x2, y2]"""
        z = np.array([
            (measurement[0] + measurement[2]) / 2.0,
            (measurement[1] + measurement[3]) / 2.0,
            measurement[2] - measurement[0],
            measurement[3] - measurement[1]
        ])
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + np.eye(self.dim_z) * 2.0
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        
    def get_state(self):
        """Get current bounding box [x1, y1, x2, y2]"""
        x_c, y_c, w, h = self.x[0], self.x[1], max(1, self.x[2]), max(1, self.x[3])
        return [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]


def compute_iou_matrix(pred_boxes, det_boxes):
    """Compute IoU matrix between predicted boxes and detections"""
    n_pred = len(pred_boxes)
    n_det = len(det_boxes)
    iou_matrix = np.zeros((n_pred, n_det))
    
    for i in range(n_pred):
        for j in range(n_det):
            p = pred_boxes[i]
            d = det_boxes[j]
            x1 = max(p[0], d[0])
            y1 = max(p[1], d[1])
            x2 = min(p[2], d[2])
            y2 = min(p[3], d[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_p = (p[2] - p[0]) * (p[3] - p[1])
            area_d = (d[2] - d[0]) * (d[3] - d[1])
            union = area_p + area_d - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0
    
    return iou_matrix


def hungarian_match(cost_matrix, threshold=None):
    """Solve assignment using scipy's linear_sum_assignment"""
    from scipy.optimize import linear_sum_assignment
    
    if cost_matrix.size == 0:
        return [], [], [], []
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched = []
    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_dets = set(range(cost_matrix.shape[1]))
    
    for r, c in zip(row_ind, col_ind):
        if threshold is not None and cost_matrix[r, c] > threshold:
            continue
        matched.append((r, c))
        unmatched_tracks.discard(r)
        unmatched_dets.discard(c)
    
    return matched, list(unmatched_tracks), list(unmatched_dets)
