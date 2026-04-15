from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0,0],
                              [0,1,0,0,0,1,0,0],
                              [0,0,1,0,0,0,1,0],
                              [0,0,0,1,0,0,0,1],
                              [0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.H = np.array([[1,0,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0]], dtype=np.float32)
        self.kf.R *= 10.  # adjust
        self.kf.P[4:,4:] *= 1000.
        self.kf.P[:4,:4] *= 1.0
        self.kf.x[:4] = self._from_xywh(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._from_xywh(bbox))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.to_xywh().copy())
        return self

    def get_state(self):
        return self.to_xywh()

    def to_xywh(self):
        ret = self.kf.x[:4].copy()
        ret[:2] += ret[2:] / 2  # xy center to tl
        return ret

    def _from_xywh(self, xywh):
        ret = np.array(xywh).copy()
        ret[:2] -= ret[2:] / 2  # tl
        return ret
