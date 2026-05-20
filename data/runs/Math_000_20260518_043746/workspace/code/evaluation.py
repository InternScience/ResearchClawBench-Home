"""
MOT Evaluation Metrics.
Standard metrics: MOTA, MOTP, IDF1, IDsw, MT, ML, FP, FN, Frag
"""
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_mot_metrics(tracker_results, gt_data, iou_threshold=0.5):
    """
    Compute MOT evaluation metrics.
    
    Args:
        tracker_results: dict {frame_idx: {track_id: bbox}}
        gt_data: list of frame dicts from the dataset
        iou_threshold: IoU threshold for a match to count as TP
        
    Returns:
        dict of metrics
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    
    # Per-GT-ID tracking
    gt_id_frames = defaultdict(lambda: defaultdict(int))  # gt_id -> {frame: matched_track}
    gt_id_frame_set = defaultdict(set)  # gt_id -> set of frames where it appears
    track_id_frame_set = defaultdict(set)  # track_id -> set of frames where it appears
    
    all_frames = sorted(gt_data.keys())
    
    for frame_idx in all_frames:
        frame_data = gt_data[frame_idx]
        gt_bboxes = frame_data['gt_bboxes']
        gt_ids = frame_data['gt_ids']
        
        # Get tracker results for this frame
        frame_tracks = tracker_results.get(frame_idx, {})
        track_ids = list(frame_tracks.keys())
        track_bboxes = [frame_tracks[tid] for tid in track_ids]
        
        n_gt = len(gt_bboxes)
        n_det = len(track_bboxes)
        total_gt += n_gt
        
        # Build cost matrix
        if n_gt == 0 or n_det == 0:
            total_fn += n_gt
            total_fp += n_det
            continue
        
        iou_matrix = np.zeros((n_gt, n_det))
        for i in range(n_gt):
            for j in range(n_det):
                iou_matrix[i, j] = compute_iou(gt_bboxes[i], track_bboxes[j])
        
        # Greedy matching
        matched_gt = set()
        matched_det = set()
        
        # Sort by IoU descending
        pairs = []
        for i in range(n_gt):
            for j in range(n_det):
                pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True)
        
        for iou_val, gt_idx, det_idx in pairs:
            if gt_idx in matched_gt or det_idx in matched_det:
                continue
            if iou_val < iou_threshold:
                continue
            
            matched_gt.add(gt_idx)
            matched_det.add(det_idx)
            total_tp += 1
            total_iou_sum += iou_val
            
            # Track ID association
            gt_id = gt_ids[gt_idx]
            track_id = track_ids[det_idx]
            gt_id_frames[gt_id][frame_idx] = track_id
            gt_id_frame_set[gt_id].add(frame_idx)
            track_id_frame_set[track_id].add(frame_idx)
        
        total_fn += n_gt - len(matched_gt)
        total_fp += n_det - len(matched_det)
    
    # Compute MOTA
    mota = 1.0 - (total_fn + total_fp + 0) / max(total_gt, 1)
    
    # Compute MOTP
    motp = total_iou_sum / max(total_tp, 1)
    
    # Compute ID switches
    id_switches = 0
    for gt_id in gt_id_frames:
        frames_sorted = sorted(gt_id_frames[gt_id].keys())
        for i in range(1, len(frames_sorted)):
            if gt_id_frames[gt_id][frames_sorted[i]] != gt_id_frames[gt_id][frames_sorted[i-1]]:
                # Check if it's a re-detection (same track appears again after gap)
                id_switches += 1
    
    # Compute Mostly Tracked (MT), Mostly Lost (ML)
    mt_count = 0
    ml_count = 0
    total_unique_gt = len(set(gt_id for frame_idx, f in gt_data.items() for gt_id in f['gt_ids']))
    
    for gt_id in gt_id_frames:
        # Count frames where this GT object appears
        total_frames_with_gt = sum(1 for f in all_frames 
                                    if gt_id in gt_data[f]['gt_ids'])
        
        # Count frames where it was successfully tracked
        tracked_frames = len(gt_id_frames[gt_id])
        
        ratio = tracked_frames / max(total_frames_with_gt, 1)
        
        if ratio >= 0.8:
            mt_count += 1
        elif ratio < 0.2:
            ml_count += 1
    
    # IDF1 - simplified
    # ID precision: correct / total detected
    # ID recall: correct / total GT
    id_precision = total_tp / max(total_tp + total_fp, 1)
    id_recall = total_tp / max(total_tp + total_fn, 1)
    idf1 = 2 * id_precision * id_recall / max(id_precision + id_recall, 1e-6)
    
    metrics = {
        'MOTA': mota * 100,
        'MOTP': motp * 100,
        'IDF1': idf1 * 100,
        'IDsw': id_switches,
        'MT': mt_count,
        'ML': ml_count,
        'FP': total_fp,
        'FN': total_fn,
        'TP': total_tp,
        'GT': total_gt,
        'Fragments': id_switches,  # Simplified
        'n_unique_gt': total_unique_gt,
        'ID_precision': id_precision * 100,
        'ID_recall': id_recall * 100
    }
    
    return metrics
