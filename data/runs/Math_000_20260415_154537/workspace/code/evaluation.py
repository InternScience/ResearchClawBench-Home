"""
Evaluation Metrics for Multi-Object Tracking
============================================

Implements MOT metrics:
- MOTA (Multi-Object Tracking Accuracy)
- MOTP (Multi-Object Tracking Precision)
- IDF1 (ID F1 Score)
- ID Switches
- Fragmentation
- Mostly Tracked (MT)
- Mostly Lost (ML)
"""

import numpy as np
from collections import defaultdict


def compute_mot_metrics(ground_truth, predictions, iou_threshold=0.5):
    """
    Compute MOT metrics.
    
    Args:
        ground_truth: Dict mapping frame_id to list of (gt_id, bbox) tuples
        predictions: Dict mapping frame_id to list of (track_id, bbox) tuples
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary of metrics
    """
    total_gt = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_fragments = 0
    
    id_mapping = {}  # Maps gt_id -> track_id for consistency check
    prev_id_mapping = {}
    
    gt_trajectories = defaultdict(list)  # gt_id -> list of (frame, matched)
    track_trajectories = defaultdict(list)  # track_id -> list of (frame, matched)
    
    all_frames = sorted(set(list(ground_truth.keys()) + list(predictions.keys())))
    
    for frame_id in all_frames:
        gt_boxes = ground_truth.get(frame_id, [])
        pred_boxes = predictions.get(frame_id, [])
        
        total_gt += len(gt_boxes)
        
        # Match predictions to ground truth
        matched_pairs = []
        matched_gt = set()
        matched_pred = set()
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, (gt_id, gt_box) in enumerate(gt_boxes):
                for j, (track_id, pred_box) in enumerate(pred_boxes):
                    iou = compute_iou(gt_box, pred_box)
                    iou_matrix[i, j] = iou
            
            # Greedy matching based on IoU
            while True:
                if iou_matrix.size == 0 or iou_matrix.max() < iou_threshold:
                    break
                
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[i, j] >= iou_threshold:
                    matched_pairs.append((i, j))
                    matched_gt.add(i)
                    matched_pred.add(j)
                    iou_matrix[i, :] = 0
                    iou_matrix[:, j] = 0
                else:
                    break
        
        # Check for ID switches
        current_id_mapping = {}
        for gt_idx, pred_idx in matched_pairs:
            gt_id = gt_boxes[gt_idx][0]
            track_id = pred_boxes[pred_idx][0]
            current_id_mapping[gt_id] = track_id
            
            # Check ID switch
            if gt_id in id_mapping and id_mapping[gt_id] != track_id:
                total_id_switches += 1
            
            # Track trajectory
            gt_trajectories[gt_id].append((frame_id, True))
            track_trajectories[track_id].append((frame_id, True))
            total_matches += 1
        
        # Count false negatives (unmatched ground truth)
        fn = len(gt_boxes) - len(matched_gt)
        total_fn += fn
        
        # Count false positives (unmatched predictions)
        fp = len(pred_boxes) - len(matched_pred)
        total_fp += fp
        
        # Track unmatched trajectories
        for i, (gt_id, _) in enumerate(gt_boxes):
            if i not in matched_gt:
                gt_trajectories[gt_id].append((frame_id, False))
        
        for j, (track_id, _) in enumerate(pred_boxes):
            if j not in matched_pred:
                track_trajectories[track_id].append((frame_id, False))
        
        # Check for fragments
        if prev_id_mapping:
            for gt_id in prev_id_mapping:
                if gt_id in current_id_mapping:
                    if prev_id_mapping[gt_id] == current_id_mapping[gt_id]:
                        # Same ID, check continuity
                        prev_track_id = prev_id_mapping[gt_id]
                        # Already counted in matches
                        pass
                else:
                    # Was tracked, now lost - potential fragment
                    pass
        
        prev_id_mapping = current_id_mapping
        id_mapping.update(current_id_mapping)
    
    # Compute MOTA
    if total_gt > 0:
        mota = 1 - (total_fn + total_fp + total_id_switches) / total_gt
    else:
        mota = 0
    
    # Compute MOTP (average IoU of matched pairs)
    # This is simplified; proper MOTP requires distance computation
    
    # Compute IDF1
    # Simplified IDF1 based on ID consistency
    idtp = total_matches
    idfp = total_fp
    idfn = total_fn
    
    if idtp + idfp > 0 and idtp + idfn > 0:
        idp = idtp / (idtp + idfp)
        idr = idtp / (idtp + idfn)
        idf1 = 2 * idp * idr / (idp + idr) if (idp + idr) > 0 else 0
    else:
        idf1 = 0
    
    # Compute MT and ML
    mt_count = 0
    ml_count = 0
    
    for gt_id, frames in gt_trajectories.items():
        matched_frames = sum(1 for _, matched in frames if matched)
        total_frames = len(frames)
        if total_frames > 0:
            ratio = matched_frames / total_frames
            if ratio >= 0.8:
                mt_count += 1
            elif ratio <= 0.2:
                ml_count += 1
    
    return {
        'MOTA': mota * 100,
        'IDF1': idf1 * 100,
        'ID_Switches': total_id_switches,
        'FP': total_fp,
        'FN': total_fn,
        'MT': mt_count,
        'ML': ml_count,
        'GT_Total': total_gt,
        'Matches': total_matches
    }


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area < 1e-6:
        return 0.0
    
    return inter_area / union_area


def compute_clear_mot_metrics(ground_truth, predictions, iou_threshold=0.5):
    """
    Compute CLEAR MOT metrics using a more accurate method.
    """
    # Track-to-GT matching history
    matches = defaultdict(list)  # frame -> list of (gt_id, track_id, iou)
    
    all_gt_ids = set()
    all_track_ids = set()
    
    for frame_id in sorted(ground_truth.keys()):
        gt_data = ground_truth[frame_id]
        pred_data = predictions.get(frame_id, [])
        
        gt_ids = [g[0] for g in gt_data]
        gt_boxes = [g[1] for g in gt_data]
        track_ids = [p[0] for p in pred_data]
        track_boxes = [p[1] for p in pred_data]
        
        all_gt_ids.update(gt_ids)
        all_track_ids.update(track_ids)
        
        # Compute matches
        if len(gt_boxes) > 0 and len(track_boxes) > 0:
            iou_matrix = np.zeros((len(gt_boxes), len(track_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, track_box in enumerate(track_boxes):
                    iou_matrix[i, j] = compute_iou(gt_box, track_box)
            
            frame_matches = []
            used_gt = set()
            used_track = set()
            
            # Greedy matching
            while len(used_gt) < len(gt_boxes) and len(used_track) < len(track_boxes):
                max_iou = 0
                max_i, max_j = -1, -1
                for i in range(len(gt_boxes)):
                    if i in used_gt:
                        continue
                    for j in range(len(track_boxes)):
                        if j in used_track:
                            continue
                        if iou_matrix[i, j] > max_iou:
                            max_iou = iou_matrix[i, j]
                            max_i, max_j = i, j
                
                if max_iou >= iou_threshold:
                    frame_matches.append((gt_ids[max_i], track_ids[max_j], max_iou))
                    used_gt.add(max_i)
                    used_track.add(max_j)
                else:
                    break
            
            matches[frame_id] = frame_matches
    
    # Compute metrics
    total_gt = sum(len(ground_truth[f]) for f in ground_truth)
    total_matches_count = sum(len(m) for m in matches.values())
    total_fp = sum(
        len(predictions.get(f, [])) - len(matches.get(f, []))
        for f in set(list(ground_truth.keys()) + list(predictions.keys()))
    )
    total_fn = total_gt - total_matches_count
    
    # Count ID switches
    prev_mapping = {}
    id_switches = 0
    
    for frame_id in sorted(matches.keys()):
        current_mapping = {}
        for gt_id, track_id, iou in matches[frame_id]:
            current_mapping[gt_id] = track_id
            
            if gt_id in prev_mapping and prev_mapping[gt_id] != track_id:
                id_switches += 1
        
        prev_mapping = current_mapping
    
    # Compute MOTA
    if total_gt > 0:
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt
    else:
        mota = 0
    
    # Compute IDF1 (simplified)
    if total_matches_count > 0:
        idf1 = total_matches_count / (total_matches_count + 0.5 * (total_fp + total_fn))
        idf1 *= 100
    else:
        idf1 = 0
    
    # Compute precision and recall
    precision = total_matches_count / (total_matches_count + total_fp) if (total_matches_count + total_fp) > 0 else 0
    recall = total_matches_count / total_gt if total_gt > 0 else 0
    
    return {
        'MOTA': mota * 100,
        'IDF1': idf1,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'ID_Switches': id_switches,
        'FP': total_fp,
        'FN': total_fn,
        'GT_Total': total_gt,
        'Matches': total_matches_count
    }
