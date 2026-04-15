"""
SparseTrack vs ByteTrack: Multi-Object Tracking with Pseudo-Depth Estimation
=============================================================================
Optimized version with adaptive thresholds for dense scenes.
"""

import json
import numpy as np
from collections import defaultdict
import os

def load_data(path):
    with open(path) as f:
        return json.load(f)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def compute_iou_matrix(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    iou_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            iou_mat[i, j] = iou(boxes1[i], boxes2[j])
    return iou_mat

def combined_similarity(box1, box2, alpha=0.5):
    iou_score = iou(box1, box2)
    cx1, cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    cx2, cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    w = (box1[2]-box1[0] + box2[2]-box2[0]) / 2
    h = (box1[3]-box1[1] + box2[3]-box2[1]) / 2
    norm_dist = dist / (np.sqrt(w*h) + 1e-6)
    dist_score = np.exp(-norm_dist / 2.0)
    return alpha * iou_score + (1 - alpha) * dist_score

def compute_similarity_matrix(boxes1, boxes2, alpha=0.5):
    n1, n2 = len(boxes1), len(boxes2)
    sim_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            sim_mat[i, j] = combined_similarity(boxes1[i], boxes2[j], alpha=alpha)
    return sim_mat

def hungarian_matching(cost_matrix):
    matches = []
    used_rows = set()
    used_cols = set()
    entries = []
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            entries.append((cost_matrix[i, j], i, j))
    entries.sort(reverse=True)
    for cost, i, j in entries:
        if i not in used_rows and j not in used_cols:
            matches.append((i, j))
            used_rows.add(i)
            used_cols.add(j)
    return matches

def estimate_pseudo_depth(bbox, img_height=600):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    y_center = (y1 + y2) / 2.0
    area = w * h
    position_factor = y_center / img_height
    pseudo_depth = (1.0 / (area + 1e-6)) * (1.0 - position_factor + 0.5)
    return pseudo_depth

def decompose_by_depth(detections, num_layers=3):
    if len(detections) == 0:
        return [[] for _ in range(num_layers)]
    depths = [estimate_pseudo_depth(d['bbox']) for d in detections]
    depth_array = np.array(depths)
    percentiles = np.linspace(0, 100, num_layers + 1)
    boundaries = [np.percentile(depth_array, p) for p in percentiles]
    layers = [[] for _ in range(num_layers)]
    for i, d in enumerate(detections):
        assigned = False
        for layer_idx in range(num_layers):
            if boundaries[layer_idx] <= depth_array[i] <= boundaries[layer_idx + 1] + 1e-9:
                layers[layer_idx].append(d)
                assigned = True
                break
        if not assigned:
            layers[-1].append(d)
    return layers

class Track:
    _next_id = 0
    def __init__(self, bbox, score, frame_id):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.score = score
        self.history = [(frame_id, bbox, score)]
        self.frames_since_update = 0
        self.age = 1
        self.hits = 1
        self.state = 'active'
    
    def update(self, bbox, score, frame_id):
        self.bbox = bbox
        self.score = score
        self.history.append((frame_id, bbox, score))
        self.frames_since_update = 0
        self.age += 1
        self.hits += 1
        self.state = 'active'
    
    def mark_missed(self):
        self.frames_since_update += 1
        self.age += 1
        if self.frames_since_update > 50:
            self.state = 'deleted'
        elif self.frames_since_update > 3:
            self.state = 'lost'
    
    def predict(self):
        if len(self.history) < 2:
            return self.bbox
        prev_bbox = self.history[-2][1]
        curr_bbox = self.history[-1][1]
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]
        damping = 0.8
        return [
            curr_bbox[0] + dx * damping,
            curr_bbox[1] + dy * damping,
            curr_bbox[2] + dx * damping,
            curr_bbox[3] + dy * damping
        ]

def reset_track_ids():
    Track._next_id = 0

class SparseTrackTracker:
    def __init__(self, high_thresh=0.3, low_thresh=0.05, sim_thresh=0.15,
                 max_age=50, num_depth_layers=4):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.sim_thresh = sim_thresh
        self.max_age = max_age
        self.num_depth_layers = num_depth_layers
        self.tracks = []
    
    def update(self, detections, frame_id):
        high_conf = [d for d in detections if d['score'] >= self.high_thresh]
        low_conf = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]
        
        depth_layers = decompose_by_depth(high_conf, self.num_depth_layers)
        
        matched_track_ids = set()
        matched_det_ids = set()
        
        for layer_idx, layer_dets in enumerate(depth_layers):
            if not layer_dets:
                continue
            
            active_tracks = [t for t in self.tracks 
                           if t.state != 'deleted' and id(t) not in matched_track_ids]
            
            if not active_tracks:
                for d in layer_dets:
                    new_track = Track(d['bbox'], d['score'], frame_id)
                    self.tracks.append(new_track)
                    matched_det_ids.add(id(d))
                continue
            
            track_bboxes = [t.predict() for t in active_tracks]
            det_bboxes = [d['bbox'] for d in layer_dets]
            
            sim_mat = compute_similarity_matrix(track_bboxes, det_bboxes, alpha=0.4)
            matches = hungarian_matching(sim_mat)
            
            for t_idx, d_idx in matches:
                if sim_mat[t_idx, d_idx] >= self.sim_thresh:
                    active_tracks[t_idx].update(layer_dets[d_idx]['bbox'],
                                               layer_dets[d_idx]['score'], frame_id)
                    matched_track_ids.add(id(active_tracks[t_idx]))
                    matched_det_ids.add(id(layer_dets[d_idx]))
            
            for d in layer_dets:
                if id(d) not in matched_det_ids:
                    new_track = Track(d['bbox'], d['score'], frame_id)
                    self.tracks.append(new_track)
                    matched_det_ids.add(id(d))
        
        if low_conf:
            unmatched_tracks = [t for t in self.tracks
                              if t.state != 'deleted' and id(t) not in matched_track_ids]
            
            if unmatched_tracks:
                track_bboxes = [t.predict() for t in unmatched_tracks]
                det_bboxes = [d['bbox'] for d in low_conf]
                sim_mat = compute_similarity_matrix(track_bboxes, det_bboxes, alpha=0.4)
                matches = hungarian_matching(sim_mat)
                
                for t_idx, d_idx in matches:
                    if sim_mat[t_idx, d_idx] >= self.sim_thresh * 0.7:
                        unmatched_tracks[t_idx].update(low_conf[d_idx]['bbox'],
                                                      low_conf[d_idx]['score'], frame_id)
                        matched_track_ids.add(id(unmatched_tracks[t_idx]))
        
        for track in self.tracks:
            if id(track) not in matched_track_ids and track.state != 'deleted':
                track.mark_missed()
        
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
        
        results = []
        for track in self.tracks:
            if track.state == 'active':
                results.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'score': track.score
                })
        return results

class ByteTrackTracker:
    def __init__(self, high_thresh=0.3, low_thresh=0.05, sim_thresh=0.15, max_age=50):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.sim_thresh = sim_thresh
        self.max_age = max_age
        self.tracks = []
    
    def update(self, detections, frame_id):
        high_conf = [d for d in detections if d['score'] >= self.high_thresh]
        low_conf = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]
        
        active_tracks = [t for t in self.tracks if t.state != 'deleted']
        matched_track_ids = set()
        matched_det_ids = set()
        
        if active_tracks and high_conf:
            track_bboxes = [t.predict() for t in active_tracks]
            det_bboxes = [d['bbox'] for d in high_conf]
            sim_mat = compute_similarity_matrix(track_bboxes, det_bboxes, alpha=0.4)
            matches = hungarian_matching(sim_mat)
            
            for t_idx, d_idx in matches:
                if sim_mat[t_idx, d_idx] >= self.sim_thresh:
                    active_tracks[t_idx].update(high_conf[d_idx]['bbox'],
                                               high_conf[d_idx]['score'], frame_id)
                    matched_track_ids.add(id(active_tracks[t_idx]))
                    matched_det_ids.add(id(high_conf[d_idx]))
        
        unmatched_tracks = [t for t in active_tracks if id(t) not in matched_track_ids]
        
        if unmatched_tracks and low_conf:
            track_bboxes = [t.predict() for t in unmatched_tracks]
            det_bboxes = [d['bbox'] for d in low_conf]
            sim_mat = compute_similarity_matrix(track_bboxes, det_bboxes, alpha=0.4)
            matches = hungarian_matching(sim_mat)
            
            for t_idx, d_idx in matches:
                if sim_mat[t_idx, d_idx] >= self.sim_thresh * 0.7:
                    unmatched_tracks[t_idx].update(low_conf[d_idx]['bbox'],
                                                  low_conf[d_idx]['score'], frame_id)
                    matched_track_ids.add(id(unmatched_tracks[t_idx]))
        
        for d in high_conf:
            if id(d) not in matched_det_ids:
                new_track = Track(d['bbox'], d['score'], frame_id)
                self.tracks.append(new_track)
        
        for track in self.tracks:
            if id(track) not in matched_track_ids and track.state != 'deleted':
                track.mark_missed()
        
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
        
        results = []
        for track in self.tracks:
            if track.state == 'active':
                results.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'score': track.score
                })
        return results

def compute_mot_metrics(trajectories, gt_data, iou_threshold=0.5):
    gt_traj = defaultdict(dict)
    for frame_data in gt_data:
        fid = frame_data['frame']
        for bbox, gt_id in zip(frame_data['gt_bboxes'], frame_data['gt_ids']):
            gt_traj[gt_id][fid] = bbox
    
    pred_traj = defaultdict(dict)
    for tid, frames in trajectories.items():
        for fid, bbox in frames:
            pred_traj[tid][fid] = bbox
    
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_matches = 0
    total_iou_sum = 0.0
    
    gt_to_pred = {}
    gt_fragment_count = defaultdict(int)
    gt_was_matched = {}
    
    for frame_data in gt_data:
        fid = frame_data['frame']
        gt_bboxes = frame_data['gt_bboxes']
        gt_ids = frame_data['gt_ids']
        
        pred_bboxes = []
        pred_ids = []
        for tid, frames_dict in pred_traj.items():
            if fid in frames_dict:
                pred_bboxes.append(frames_dict[fid])
                pred_ids.append(tid)
        
        total_gt += len(gt_bboxes)
        
        if not pred_bboxes:
            total_fn += len(gt_bboxes)
            for gid in gt_ids:
                if gt_was_matched.get(gid, False):
                    gt_fragment_count[gid] += 1
                gt_was_matched[gid] = False
            continue
        
        iou_mat = compute_iou_matrix(gt_bboxes, pred_bboxes)
        matches = hungarian_matching(iou_mat)
        
        matched_gt = set()
        matched_pred = set()
        
        for g_idx, p_idx in matches:
            if iou_mat[g_idx, p_idx] >= iou_threshold:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                total_matches += 1
                total_iou_sum += iou_mat[g_idx, p_idx]
                
                gt_id = gt_ids[g_idx]
                pred_id = pred_ids[p_idx]
                
                if gt_id in gt_to_pred and gt_to_pred[gt_id] != pred_id:
                    total_id_switches += 1
                gt_to_pred[gt_id] = pred_id
                
                if not gt_was_matched.get(gt_id, False):
                    gt_was_matched[gt_id] = True
        
        total_fn += len(gt_bboxes) - len(matched_gt)
        total_fp += len(pred_bboxes) - len(matched_pred)
        
        for g_idx, gt_id in enumerate(gt_ids):
            if g_idx not in matched_gt:
                if gt_was_matched.get(gt_id, False):
                    gt_fragment_count[gt_id] += 1
                gt_was_matched[gt_id] = False
    
    total_fragments = sum(gt_fragment_count.values())
    
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)
    motp = total_iou_sum / max(total_matches, 1)
    
    id_precision = total_matches / max(total_matches + total_fp, 1)
    id_recall = total_matches / max(total_matches + total_fn, 1)
    idf1 = 2 * id_precision * id_recall / max(id_precision + id_recall, 1e-6)
    
    return {
        'MOTA': mota,
        'MOTP': motp,
        'IDF1': idf1,
        'ID_switches': total_id_switches,
        'Fragments': total_fragments,
        'FP': total_fp,
        'FN': total_fn,
        'Matches': total_matches,
        'GT': total_gt
    }

def compute_occlusion_metrics(trajectories, gt_data):
    occlusion_levels = defaultdict(list)
    
    for frame_data in gt_data:
        fid = frame_data['frame']
        gt_bboxes = frame_data['gt_bboxes']
        gt_ids = frame_data['gt_ids']
        
        for i, (bbox_i, gid_i) in enumerate(zip(gt_bboxes, gt_ids)):
            overlap_count = 0
            for j, (bbox_j, gid_j) in enumerate(zip(gt_bboxes, gt_ids)):
                if i != j and iou(bbox_i, bbox_j) > 0.05:
                    overlap_count += 1
            
            tracked = False
            for tid, frames in trajectories.items():
                for f, b in frames:
                    if f == fid and iou(bbox_i, b) >= 0.5:
                        tracked = True
                        break
                if tracked:
                    break
            
            level = min(overlap_count // 10 * 10, 90)
            occlusion_levels[level].append(1 if tracked else 0)
    
    results = {}
    for level in sorted(occlusion_levels.keys()):
        tracking_rate = np.mean(occlusion_levels[level])
        count = len(occlusion_levels[level])
        results[level] = {'tracking_rate': tracking_rate, 'count': count}
    
    return results

def compute_per_object_recall(trajectories, gt_data):
    gt_traj = defaultdict(dict)
    for frame_data in gt_data:
        fid = frame_data['frame']
        for bbox, gt_id in zip(frame_data['gt_bboxes'], frame_data['gt_ids']):
            gt_traj[gt_id][fid] = bbox
    
    pred_traj = defaultdict(dict)
    for tid, frames in trajectories.items():
        for fid, bbox in frames:
            pred_traj[tid][fid] = bbox
    
    per_object_recall = {}
    for gt_id, gt_frames in gt_traj.items():
        matched = 0
        for fid, gt_bbox in gt_frames.items():
            for tid, pred_frames in pred_traj.items():
                if fid in pred_frames and iou(gt_bbox, pred_frames[fid]) >= 0.5:
                    matched += 1
                    break
        per_object_recall[gt_id] = matched / len(gt_frames) if gt_frames else 0
    
    return per_object_recall

def run_tracker(tracker_class, tracker_name, data, **kwargs):
    reset_track_ids()
    tracker = tracker_class(**kwargs)
    trajectories = defaultdict(list)
    
    for frame_data in data:
        fid = frame_data['frame']
        detections = frame_data['detections']
        results = tracker.update(detections, fid)
        for r in results:
            trajectories[r['track_id']].append((fid, r['bbox']))
    
    metrics = compute_mot_metrics(trajectories, data)
    occlusion_metrics = compute_occlusion_metrics(trajectories, data)
    per_object_recall = compute_per_object_recall(trajectories, data)
    
    return trajectories, metrics, occlusion_metrics, per_object_recall

def main():
    data_path = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_000_20260414_091314/data/simulated_sequence.json'
    output_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_000_20260414_091314/outputs'
    images_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_000_20260414_091314/report/images'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print("Loading data...")
    data = load_data(data_path)
    print(f"Loaded {len(data)} frames")
    
    print("\nRunning SparseTrack...")
    sparse_traj, sparse_metrics, sparse_occlusion, sparse_recall = run_tracker(
        SparseTrackTracker, 'SparseTrack', data,
        high_thresh=0.3, low_thresh=0.05, sim_thresh=0.15, max_age=50, num_depth_layers=4
    )
    print(f"SparseTrack: MOTA={sparse_metrics['MOTA']:.4f}, IDF1={sparse_metrics['IDF1']:.4f}, "
          f"IDSw={sparse_metrics['ID_switches']}, Tracks={len(sparse_traj)}")
    
    print("\nRunning ByteTrack...")
    byte_traj, byte_metrics, byte_occlusion, byte_recall = run_tracker(
        ByteTrackTracker, 'ByteTrack', data,
        high_thresh=0.3, low_thresh=0.05, sim_thresh=0.15, max_age=50
    )
    print(f"ByteTrack: MOTA={byte_metrics['MOTA']:.4f}, IDF1={byte_metrics['IDF1']:.4f}, "
          f"IDSw={byte_metrics['ID_switches']}, Tracks={len(byte_traj)}")
    
    results = {
        'SparseTrack': {
            'metrics': sparse_metrics,
            'occlusion_metrics': {str(k): v for k, v in sparse_occlusion.items()},
            'num_tracks': len(sparse_traj),
            'avg_track_length': float(np.mean([len(v) for v in sparse_traj.values()])) if sparse_traj else 0,
            'per_object_recall_mean': float(np.mean(list(sparse_recall.values()))),
            'per_object_recall_std': float(np.std(list(sparse_recall.values())))
        },
        'ByteTrack': {
            'metrics': byte_metrics,
            'occlusion_metrics': {str(k): v for k, v in byte_occlusion.items()},
            'num_tracks': len(byte_traj),
            'avg_track_length': float(np.mean([len(v) for v in byte_traj.values()])) if byte_traj else 0,
            'per_object_recall_mean': float(np.mean(list(byte_recall.values()))),
            'per_object_recall_std': float(np.std(list(byte_recall.values())))
        }
    }
    
    with open(f'{output_dir}/tracking_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    recall_data = {
        'SparseTrack': {str(k): v for k, v in sparse_recall.items()},
        'ByteTrack': {str(k): v for k, v in byte_recall.items()}
    }
    with open(f'{output_dir}/per_object_recall.json', 'w') as f:
        json.dump(recall_data, f, indent=2)
    
    traj_export = {}
    for name, traj in [('SparseTrack', sparse_traj), ('ByteTrack', byte_traj)]:
        traj_export[name] = {str(k): [(f, b) for f, b in v] for k, v in traj.items()}
    with open(f'{output_dir}/trajectories.json', 'w') as f:
        json.dump(traj_export, f, indent=2)
    
    print("\nResults saved.")
    return data, sparse_traj, byte_traj, sparse_metrics, byte_metrics, sparse_occlusion, byte_occlusion, sparse_recall, byte_recall

if __name__ == '__main__':
    main()
