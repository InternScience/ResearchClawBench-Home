import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class KalmanFilter:
    def __init__(self, bbox):
        self.dt = 1
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self.dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self.dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        self.P = np.eye(8) * 10
        self.Q = np.eye(8) * 0.1
        self.R = np.eye(4) * 1
        
        self.x = np.zeros((8, 1))
        self.x[:4, 0] = bbox
        
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:4, 0]
        
    def update(self, z):
        z = np.array(z).reshape((4, 1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(8)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.kf = KalmanFilter(bbox)
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.state = 'active'
        
    def predict(self):
        bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return bbox
        
    def update(self, bbox):
        self.kf.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.state = 'active'
        
    def get_bbox(self):
        return self.kf.x[:4, 0]

