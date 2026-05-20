import sys, json
sys.path.insert(0, '.')

# Minimal imports
from code.kalman_filter import KalmanFilter, compute_iou_matrix, hungarian_match

with open('data/simulated_sequence.json') as f:
    data = json.load(f)

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        
    def process_frame(self, detections, frame_idx):
        pred_boxes = []
        for track in self.tracks:
            track['kf'].predict()
            pred_boxes.append(track['kf'].get_state())
        
        det_boxes = [d['bbox'] for d in detections]
        
        if pred_boxes and det_boxes:
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes)
            cost = 1.0 - iou_mat
            matched, unmatched_tracks, unmatched_dets = hungarian_match(
                cost, threshold=1.0 - self.iou_thresh
            )
            for t_idx, d_idx in matched:
                self.tracks[t_idx]['kf'].update(detections[d_idx]['bbox'])
                self.tracks[t_idx]['time_since_update'] = 0
                self.tracks[t_idx]['hits'] += 1
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
        
        for t_idx in unmatched_tracks:
            self.tracks[t_idx]['time_since_update'] += 1
        
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        for d_idx in unmatched_dets:
            kf = KalmanFilter()
            kf.initiate(detections[d_idx]['bbox'])
            self.tracks.append({'id': self.next_id, 'kf': kf, 'time_since_update': 0, 'hits': 1})
            self.next_id += 1
        
        results = {}
        for track in self.tracks:
            if track['time_since_update'] == 0:
                results[track['id']] = track['kf'].get_state()
        return results

tracker = SimpleTracker(iou_thresh=0.3, max_age=30)
sort_results = {}
for frame in data:
    result = tracker.process_frame(frame['detections'], frame['frame'])
    sort_results[frame['frame']] = result

with open('outputs/tracking_results.json') as f:
    tracking = json.load(f)

tracking['sort'] = {str(k): {str(tid): bbox for tid, bbox in v.items()} 
                    for k, v in sort_results.items()}

with open('outputs/tracking_results.json', 'w') as f:
    json.dump(tracking, f)
print(f'SORT results saved, {tracker.next_id - 1} tracks created')
