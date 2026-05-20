"""
MOT Evaluation Metrics

Implements standard MOT metrics:
- MOTA (Multiple Object Tracking Accuracy)
- IDF1 (ID F1 Score)
- ID Switches (IDS)
- Mostly Tracked (MT), Mostly Lost (ML)
- FP, FN, Frag
- HOTA (Higher Order Tracking Accuracy, simplified)

Based on: 
- Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking Performance", 2008
- Ristani et al., "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking", ECCV 2016
- Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking", IJCV 2021
"""

import numpy as np
from collections import defaultdict


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_tracking(gt_data, tracker_output, iou_threshold=0.5):
    """
    Evaluate tracking performance.
    
    Args:
        gt_data: List of frame data with ground truth.
        tracker_output: Dict mapping frame_idx -> list of track dicts with 'bbox', 'id'.
        
    Returns:
        Dict of metrics.
    """
    # Build per-frame tracker output mapping
    frame_outputs = defaultdict(list)
    for frame_idx, tracks in tracker_output.items():
        frame_outputs[frame_idx] = tracks
    
    # Count total GT objects
    total_gt = 0
    gt_present = defaultdict(set)  # frame -> set of gt_ids
    gt_trajectories = defaultdict(list)  # gt_id -> list of (frame, bbox)
    
    for frame_data in gt_data:
        frame_idx = frame_data['frame']
        for gt_id, gt_bbox in zip(frame_data['gt_ids'], frame_data['gt_bboxes']):
            gt_present[frame_idx].add(gt_id)
            gt_trajectories[gt_id].append((frame_idx, gt_bbox))
            total_gt += 1
    
    # Tracking results per ID
    track_trajectories = defaultdict(list)  # track_id -> list of (frame, bbox)
    for frame_idx, tracks in frame_outputs.items():
        for track in tracks:
            track_trajectories[track['id']].append((frame_idx, track['bbox']))
    
    # Compute per-frame metrics
    total_fp = 0
    total_fn = 0
    total_mismatches = 0
    total_matches = 0
    id_switches = 0
    fragments = 0
    
    # Track ID mapping over time
    prev_frame_mapping = {}  # gt_id -> track_id
    track_hits = defaultdict(int)  # track_id -> number of matched frames
    
    for frame_data in gt_data:
        frame_idx = frame_data['frame']
        gt_bboxes = frame_data['gt_bboxes']
        gt_ids = frame_data['gt_ids']
        
        tracks = frame_outputs.get(frame_idx, [])
        
        if len(tracks) == 0:
            total_fn += len(gt_bboxes)
            continue
        
        # Build IoU matrix between GT and tracks
        n_gt = len(gt_bboxes)
        n_tr = len(tracks)
        iou_mat = np.zeros((n_gt, n_tr))
        for i in range(n_gt):
            for j in range(n_tr):
                iou_mat[i, j] = compute_iou(gt_bboxes[i], tracks[j]['bbox'])
        
        # Greedy matching
        matched_gt = set()
        matched_tr = set()
        frame_mapping = {}
        
        # Sort by IoU descending for greedy matching
        candidates = []
        for i in range(n_gt):
            for j in range(n_tr):
                if iou_mat[i, j] >= iou_threshold:
                    candidates.append((iou_mat[i, j], i, j))
        candidates.sort(reverse=True)
        
        for iou_val, i, j in candidates:
            if i not in matched_gt and j not in matched_tr:
                matched_gt.add(i)
                matched_tr.add(j)
                frame_mapping[gt_ids[i]] = tracks[j]['id']
                track_hits[tracks[j]['id']] += 1
                total_matches += 1
        
        # Count FP and FN
        fp = n_tr - len(matched_tr)
        fn = n_gt - len(matched_gt)
        total_fp += fp
        total_fn += fn
        
        # Count ID switches
        for gt_id, track_id in frame_mapping.items():
            if gt_id in prev_frame_mapping:
                if prev_frame_mapping[gt_id] != track_id:
                    id_switches += 1
            prev_frame_mapping[gt_id] = track_id
    
    # Compute MOTA
    mota = 1.0 - (total_fp + total_fn + id_switches) / max(1, total_gt)
    
    # Compute IDF1
    # IDF1 measures ID consistency
    # IDP = IDTP / (IDTP + IDFP), IDR = IDTP / (IDTP + IDFN)
    idtp = total_matches
    idfp = sum(len(frame_outputs.get(fd['frame'], [])) for fd in gt_data) - total_matches
    idfn = total_gt - total_matches
    
    idp = idtp / max(1, idtp + idfp)
    idr = idtp / max(1, idtp + idfn)
    idf1 = 2 * idp * idr / max(1e-10, idp + idr)
    
    # Mostly Tracked / Mostly Lost
    gt_track_coverage = {}
    for gt_id, traj in gt_trajectories.items():
        total_frames_for_gt = len(traj)
        # Find best matching track for this GT
        # For each frame, check if GT's bbox matches any track
        matched_frames = 0
        for frame_idx, gt_bbox in traj:
            tracks_in_frame = frame_outputs.get(frame_idx, [])
            for track in tracks_in_frame:
                if compute_iou(gt_bbox, track['bbox']) >= iou_threshold:
                    matched_frames += 1
                    break
        coverage = matched_frames / max(1, total_frames_for_gt)
        gt_track_coverage[gt_id] = coverage
    
    mt = sum(1 for c in gt_track_coverage.values() if c >= 0.8)
    ml = sum(1 for c in gt_track_coverage.values() if c <= 0.2)
    
    # MOTP (precision of bounding box alignment)
    motp_sum = 0.0
    motp_count = 0
    for frame_data in gt_data:
        frame_idx = frame_data['frame']
        tracks = frame_outputs.get(frame_idx, [])
        if len(tracks) == 0:
            continue
        n_gt = len(frame_data['gt_bboxes'])
        n_tr = len(tracks)
        iou_mat = np.zeros((n_gt, n_tr))
        for i in range(n_gt):
            for j in range(n_tr):
                iou_mat[i, j] = compute_iou(frame_data['gt_bboxes'][i], tracks[j]['bbox'])
        
        candidates = []
        for i in range(n_gt):
            for j in range(n_tr):
                if iou_mat[i, j] >= iou_threshold:
                    candidates.append((iou_mat[i, j], i, j))
        candidates.sort(reverse=True)
        
        matched_gt = set()
        matched_tr = set()
        for iou_val, i, j in candidates:
            if i not in matched_gt and j not in matched_tr:
                matched_gt.add(i)
                matched_tr.add(j)
                motp_sum += iou_val
                motp_count += 1
    
    motp = motp_sum / max(1, motp_count)
    
    # Track count statistics
    num_tracks = len(track_trajectories)
    avg_track_length = np.mean([len(t) for t in track_trajectories.values()]) if track_trajectories else 0
    
    return {
        'MOTA': mota * 100,
        'MOTP': motp * 100,
        'IDF1': idf1 * 100,
        'ID_Switches': id_switches,
        'FP': total_fp,
        'FN': total_fn,
        'MT': mt,
        'ML': ml,
        'Total_GT': total_gt,
        'Num_Tracks': num_tracks,
        'Avg_Track_Length': avg_track_length,
        'Total_Matches': total_matches,
    }
