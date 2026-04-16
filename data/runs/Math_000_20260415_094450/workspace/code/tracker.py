"""
Multi-Object Tracking implementations: SORT, ByteTrack, and SparseTrack.

This module implements three tracking algorithms for comparison:
1. SORT: Simple Online and Realtime Tracking (Bewley et al., 2016)
2. ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al., 2022)
3. SparseTrack: Pseudo-depth hierarchical association for dense occlusion handling
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union == 0:
        return 0.0
    return inter / union


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU matrix between two sets of boxes."""
    n = len(boxes_a)
    m = len(boxes_b)
    iou_mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            iou_mat[i, j] = compute_iou(boxes_a[i], boxes_b[j])
    return iou_mat


def hungarian_matching(cost_matrix, threshold=1.0):
    """
    Perform Hungarian matching with threshold.
    Returns matched pairs, unmatched_a, unmatched_b.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matched.append((r, c))
            unmatched_a.remove(r)
            unmatched_b.remove(c)

    return matched, unmatched_a, unmatched_b


class KalmanBoxTracker:
    """Simple Kalman filter tracker for a single object using bbox state."""

    count = 0

    def __init__(self, bbox, score=1.0):
        """Initialize tracker with bounding box [x1, y1, x2, y2]."""
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.bbox = list(bbox)
        self.score = score
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        # State: [cx, cy, w, h, vcx, vcy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        # Covariance
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty for velocities
        # Process noise
        self.Q = np.eye(8)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01
        # Measurement noise
        self.R = np.eye(4) * 1.0

    def predict(self):
        """Predict next state."""
        dt = 1.0
        F = np.eye(8)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

        # Update bbox from state
        cx, cy, w, h = self.state[:4]
        w = max(w, 1)
        h = max(h, 1)
        self.bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.bbox

    def update(self, bbox, score=1.0):
        """Update state with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score

        # Measurement
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h])

        # Kalman update
        H = np.zeros((4, 8))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1

        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

        # Update bbox
        cx, cy, w, h = self.state[:4]
        w = max(w, 1)
        h = max(h, 1)
        self.bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def get_state(self):
        """Return current bbox estimate."""
        return self.bbox


class SORTTracker:
    """SORT: Simple Online and Realtime Tracking."""

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1, score_threshold=0.0):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.score_threshold = score_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        Update tracker with detections for current frame.
        detections: list of dicts with 'bbox' and 'score' keys
        Returns: list of (track_id, bbox) pairs
        """
        self.frame_count += 1

        # Filter detections by score threshold
        det_filtered = [d for d in detections if d['score'] >= self.score_threshold]

        # Predict existing trackers
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if any(np.isnan(pred)) or any(np.isinf(pred)):
                to_remove.append(i)
            else:
                predicted.append(pred)
        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)

        # Match detections to trackers
        if len(predicted) > 0 and len(det_filtered) > 0:
            det_bboxes = [d['bbox'] for d in det_filtered]
            iou_mat = compute_iou_matrix(predicted, det_bboxes)
            cost_mat = 1.0 - iou_mat
            matched, unmatched_trk, unmatched_det = hungarian_matching(
                cost_mat, threshold=1.0 - self.iou_threshold
            )

            # Update matched trackers
            for trk_idx, det_idx in matched:
                self.trackers[trk_idx].update(
                    det_filtered[det_idx]['bbox'],
                    det_filtered[det_idx].get('score', 1.0)
                )

            # Create new trackers for unmatched detections
            for det_idx in unmatched_det:
                trk = KalmanBoxTracker(
                    det_filtered[det_idx]['bbox'],
                    det_filtered[det_idx].get('score', 1.0)
                )
                self.trackers.append(trk)
        elif len(det_filtered) > 0:
            for det in det_filtered:
                trk = KalmanBoxTracker(det['bbox'], det.get('score', 1.0))
                self.trackers.append(trk)

        # Remove dead trackers and collect results
        results = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            if trk.time_since_update > self.max_age:
                to_remove.append(i)
                continue
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append((trk.id, trk.get_state()))
        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)

        return results


class ByteTrackTracker:
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box."""

    def __init__(self, score_threshold=0.5, low_score_threshold=0.1,
                 iou_threshold=0.3, max_age=30, min_hits=1):
        self.score_threshold = score_threshold
        self.low_score_threshold = low_score_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        Update tracker with BYTE association strategy.
        detections: list of dicts with 'bbox' and 'score' keys
        Returns: list of (track_id, bbox) pairs
        """
        self.frame_count += 1

        # Split detections into high and low score
        high_det = [d for d in detections if d['score'] >= self.score_threshold]
        low_det = [d for d in detections if self.low_score_threshold <= d['score'] < self.score_threshold]

        # Predict existing trackers
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if any(np.isnan(pred)) or any(np.isinf(pred)):
                to_remove.append(i)
            else:
                predicted.append(pred)
        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)

        # First association: high score detections with all trackers
        matched_first = []
        unmatched_trk = list(range(len(self.trackers)))
        unmatched_high_det = list(range(len(high_det)))

        if len(predicted) > 0 and len(high_det) > 0:
            det_bboxes = [d['bbox'] for d in high_det]
            iou_mat = compute_iou_matrix(predicted, det_bboxes)
            cost_mat = 1.0 - iou_mat
            matched_first, unmatched_trk, unmatched_high_det = hungarian_matching(
                cost_mat, threshold=1.0 - self.iou_threshold
            )

            for trk_idx, det_idx in matched_first:
                self.trackers[trk_idx].update(
                    high_det[det_idx]['bbox'],
                    high_det[det_idx].get('score', 1.0)
                )

        # Second association: unmatched trackers with low score detections
        if len(unmatched_trk) > 0 and len(low_det) > 0:
            unmatched_trk_boxes = [predicted[i] for i in unmatched_trk]
            low_det_bboxes = [d['bbox'] for d in low_det]
            iou_mat2 = compute_iou_matrix(unmatched_trk_boxes, low_det_bboxes)
            cost_mat2 = 1.0 - iou_mat2
            matched_second, still_unmatched_trk, unmatched_low_det = hungarian_matching(
                cost_mat2, threshold=1.0 - self.iou_threshold
            )

            for local_idx, det_idx in matched_second:
                global_trk_idx = unmatched_trk[local_idx]
                self.trackers[global_trk_idx].update(
                    low_det[det_idx]['bbox'],
                    low_det[det_idx].get('score', 1.0)
                )
        else:
            still_unmatched_trk = list(range(len(unmatched_trk)))

        # Initialize new tracks from unmatched high-score detections
        for det_idx in unmatched_high_det:
            trk = KalmanBoxTracker(
                high_det[det_idx]['bbox'],
                high_det[det_idx].get('score', 1.0)
            )
            self.trackers.append(trk)

        # Collect results
        results = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            if trk.time_since_update > self.max_age:
                to_remove.append(i)
                continue
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append((trk.id, trk.get_state()))
        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)

        return results


class SparseTrackTracker:
    """
    SparseTrack: Pseudo-depth hierarchical association for dense occlusion handling.
    
    Key idea: Decompose dense target sets into sparse subsets via pseudo-depth 
    estimation, then perform hierarchical association within each depth layer.
    Objects closer to the camera (larger bbox area, lower y-center) are tracked 
    first, and their occupied regions are masked out before tracking objects at 
    deeper depth layers.
    
    The hierarchical approach reduces the ambiguity in data association by 
    resolving easier (closer, less occluded) targets first, then using that 
    information to constrain the association of more heavily occluded targets.
    """

    def __init__(self, n_depth_layers=3, score_threshold=0.5, low_score_threshold=0.1,
                 iou_threshold=0.3, max_age=30, min_hits=1, overlap_threshold=0.3):
        self.n_depth_layers = n_depth_layers
        self.score_threshold = score_threshold
        self.low_score_threshold = low_score_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.overlap_threshold = overlap_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def _estimate_pseudo_depth(self, bbox):
        """
        Estimate pseudo-depth from bounding box.
        Larger boxes and lower y-center (closer to bottom of image) suggest 
        closer distance to camera (smaller depth value).
        
        Returns depth value (0 = closest, higher = farther)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = w * h
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Depth estimation: primarily based on bbox area (larger = closer)
        # Secondary: y-center position (lower in image = closer in typical perspective)
        # Normalize area to [0, 1] range approximately
        depth = -np.log(max(area, 1.0) + 1.0)
        return depth

    def _assign_depth_layer(self, depth_values, n_layers):
        """Assign objects to depth layers based on their depth values."""
        if len(depth_values) == 0:
            return []
        
        depth_arr = np.array(depth_values)
        
        # Use quantile-based assignment for balanced layers
        layers = np.zeros(len(depth_values), dtype=int)
        if len(depth_values) <= n_layers:
            # Fewer objects than layers
            sorted_indices = np.argsort(depth_arr)
            for rank, idx in enumerate(sorted_indices):
                layers[idx] = min(rank * n_layers // len(depth_values), n_layers - 1)
        else:
            percentiles = np.linspace(0, 100, n_layers + 1)
            boundaries = np.percentile(depth_arr, percentiles)
            
            for i in range(len(depth_values)):
                assigned = False
                for l in range(n_layers):
                    if depth_arr[i] <= boundaries[l + 1] + 1e-9:
                        layers[i] = l
                        assigned = True
                        break
                if not assigned:
                    layers[i] = n_layers - 1
        
        return layers.tolist()

    def update(self, detections):
        """
        Update tracker with SparseTrack hierarchical association.
        
        Process:
        1. Estimate pseudo-depth for all detections
        2. Assign detections to depth layers
        3. For each layer (closest first):
           a. Match high-score detections to trackers in same layer
           b. Match unmatched trackers with low-score detections
           c. Record matched regions to inform deeper layers
        4. Initialize new tracks from unmatched high-score detections
        """
        self.frame_count += 1

        # Predict existing trackers
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if any(np.isnan(pred)) or any(np.isinf(pred)):
                to_remove.append(i)
            else:
                predicted.append(pred)
        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)

        if len(detections) == 0:
            results = []
            to_remove = []
            for i, trk in enumerate(self.trackers):
                if trk.time_since_update > self.max_age:
                    to_remove.append(i)
                    continue
                if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    results.append((trk.id, trk.get_state()))
            for i in sorted(to_remove, reverse=True):
                self.trackers.pop(i)
            return results

        # Estimate pseudo-depth for all detections
        det_bboxes = [d['bbox'] for d in detections]
        depth_values = [self._estimate_pseudo_depth(b) for b in det_bboxes]
        det_layers = self._assign_depth_layer(depth_values, self.n_depth_layers)

        # Assign depth layers to existing trackers based on predicted bboxes
        if len(predicted) > 0:
            trk_depth_values = [self._estimate_pseudo_depth(b) for b in predicted]
            trk_layers = self._assign_depth_layer(trk_depth_values, self.n_depth_layers)
        else:
            trk_layers = []

        # Hierarchical association: process from closest (layer 0) to farthest
        all_matched_trk = set()
        all_matched_det = set()

        for layer in range(self.n_depth_layers):
            # Get trackers in this layer (not yet matched)
            layer_trk_indices = [i for i in range(len(self.trackers))
                                 if i not in all_matched_trk and
                                 (i < len(trk_layers) and trk_layers[i] == layer)]
            
            # Get detections in this layer (not yet matched)
            layer_det_indices = [i for i in range(len(detections))
                                 if i not in all_matched_det and det_layers[i] == layer]

            if len(layer_trk_indices) == 0 and len(layer_det_indices) == 0:
                continue

            # Split detections into high and low score
            layer_det_high = [i for i in layer_det_indices 
                              if detections[i]['score'] >= self.score_threshold]
            layer_det_low = [i for i in layer_det_indices 
                             if self.low_score_threshold <= detections[i]['score'] < self.score_threshold]

            # First match: high score detections with trackers in this layer
            matched_in_layer = set()
            if len(layer_trk_indices) > 0 and len(layer_det_high) > 0:
                trk_boxes = [predicted[i] for i in layer_trk_indices]
                det_boxes = [det_bboxes[i] for i in layer_det_high]
                iou_mat = compute_iou_matrix(trk_boxes, det_boxes)
                cost_mat = 1.0 - iou_mat
                matched, um_trk, um_det = hungarian_matching(
                    cost_mat, threshold=1.0 - self.iou_threshold
                )
                for local_trk, local_det in matched:
                    global_trk = layer_trk_indices[local_trk]
                    global_det = layer_det_high[local_det]
                    self.trackers[global_trk].update(
                        detections[global_det]['bbox'],
                        detections[global_det].get('score', 1.0)
                    )
                    all_matched_trk.add(global_trk)
                    all_matched_det.add(global_det)
                    matched_in_layer.add(local_trk)

            # Second match: unmatched trackers with low score detections in this layer
            remaining_trk = [layer_trk_indices[i] for i in range(len(layer_trk_indices))
                             if i not in matched_in_layer and layer_trk_indices[i] not in all_matched_trk]
            if len(remaining_trk) > 0 and len(layer_det_low) > 0:
                trk_boxes = [predicted[i] for i in remaining_trk]
                det_boxes = [det_bboxes[i] for i in layer_det_low]
                iou_mat = compute_iou_matrix(trk_boxes, det_boxes)
                cost_mat = 1.0 - iou_mat
                matched2, _, _ = hungarian_matching(
                    cost_mat, threshold=1.0 - self.iou_threshold
                )
                for local_trk, local_det in matched2:
                    global_trk = remaining_trk[local_trk]
                    global_det = layer_det_low[local_det]
                    self.trackers[global_trk].update(
                        detections[global_det]['bbox'],
                        detections[global_det].get('score', 1.0)
                    )
                    all_matched_trk.add(global_trk)
                    all_matched_det.add(global_det)

        # Cross-layer rescue: try to match remaining unmatched trackers with 
        # unmatched detections from ANY layer (using IoU only, no depth constraint)
        remaining_trk_all = [i for i in range(len(self.trackers)) if i not in all_matched_trk]
        remaining_det_all = [i for i in range(len(detections)) if i not in all_matched_det]
        
        if len(remaining_trk_all) > 0 and len(remaining_det_all) > 0:
            # Only use high and medium score detections for cross-layer rescue
            rescue_det = [i for i in remaining_det_all if detections[i]['score'] >= self.low_score_threshold]
            if len(rescue_det) > 0:
                trk_boxes = [predicted[i] for i in remaining_trk_all]
                det_boxes = [det_bboxes[i] for i in rescue_det]
                iou_mat = compute_iou_matrix(trk_boxes, det_boxes)
                cost_mat = 1.0 - iou_mat
                matched3, _, _ = hungarian_matching(
                    cost_mat, threshold=1.0 - self.iou_threshold
                )
                for local_trk, local_det in matched3:
                    global_trk = remaining_trk_all[local_trk]
                    global_det = rescue_det[local_det]
                    self.trackers[global_trk].update(
                        detections[global_det]['bbox'],
                        detections[global_det].get('score', 1.0)
                    )
                    all_matched_trk.add(global_trk)
                    all_matched_det.add(global_det)

        # Initialize new tracks from unmatched high-score detections
        for i in range(len(detections)):
            if i not in all_matched_det and detections[i]['score'] >= self.score_threshold:
                trk = KalmanBoxTracker(
                    detections[i]['bbox'],
                    detections[i].get('score', 1.0)
                )
                self.trackers.append(trk)

        # Collect results
        results = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            if trk.time_since_update > self.max_age:
                to_remove.append(i)
                continue
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append((trk.id, trk.get_state()))
        for i in sorted(to_remove, reverse=True):
            self.trackers.pop(i)

        return results


def compute_mot_metrics(trajectories, gt_data, iou_threshold=0.5):
    """
    Compute standard MOT metrics: MOTA, MOTP, IDF1, ID Switches, etc.
    
    trajectories: dict mapping track_id -> list of (frame, bbox)
    gt_data: list of frame dicts with 'gt_bboxes' and 'gt_ids'
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_iou = 0.0
    total_matches = 0

    # Build gt per frame
    gt_per_frame = {}
    for frame_data in gt_data:
        f = frame_data['frame']
        gt_per_frame[f] = list(zip(frame_data['gt_bboxes'], frame_data['gt_ids']))

    # Build trajectory per frame
    traj_per_frame = defaultdict(list)
    for tid, frames_bboxes in trajectories.items():
        for f, bbox in frames_bboxes:
            traj_per_frame[f].append((tid, bbox))

    # Track ID mapping for ID switch detection
    gt_to_track = {}  # gt_id -> track_id mapping from previous frame

    # For IDF1 computation
    idtp_total = 0
    idfn_total = 0
    idfp_total = 0

    # For ID-based IDF1: need to compute IDTP/IDFN/IDFP based on ID consistency
    # Build mapping of gt_id -> track_id across all frames
    gt_track_matches = defaultdict(lambda: defaultdict(int))  # gt_id -> {trk_id: count}
    
    for f in sorted(gt_per_frame.keys()):
        gt_items = gt_per_frame[f]
        trk_items = traj_per_frame.get(f, [])

        total_gt += len(gt_items)

        if len(gt_items) == 0:
            total_fp += len(trk_items)
            continue
        if len(trk_items) == 0:
            total_fn += len(gt_items)
            continue

        # Compute IoU matrix
        gt_bboxes = [g[0] for g in gt_items]
        gt_ids = [g[1] for g in gt_items]
        trk_bboxes = [t[1] for t in trk_items]
        trk_ids = [t[0] for t in trk_items]

        iou_mat = compute_iou_matrix(gt_bboxes, trk_bboxes)
        cost_mat = 1.0 - iou_mat

        matched, unmatched_gt, unmatched_trk = hungarian_matching(
            cost_mat, threshold=1.0 - iou_threshold
        )

        total_tp += len(matched)
        total_fp += len(unmatched_trk)
        total_fn += len(unmatched_gt)

        for gt_idx, trk_idx in matched:
            total_iou += iou_mat[gt_idx, trk_idx]
            total_matches += 1

            # Check ID switch
            gt_id = gt_ids[gt_idx]
            trk_id = trk_ids[trk_idx]

            if gt_id in gt_to_track and gt_to_track[gt_id] != trk_id:
                total_idsw += 1

            gt_to_track[gt_id] = trk_id
            
            # Track ID matches for IDF1
            gt_track_matches[gt_id][trk_id] += 1

        # IDF1 computation
        idfn_total += len(unmatched_gt)
        idfp_total += len(unmatched_trk)

    # Compute IDTP: for each gt_id, find the best matching track_id
    idtp_total = 0
    for gt_id, trk_counts in gt_track_matches.items():
        if trk_counts:
            best_trk_id = max(trk_counts, key=trk_counts.get)
            idtp_total += trk_counts[best_trk_id]
            # IDFN: frames where this gt_id was present but not matched to best_trk_id
            # Already counted in total_fn for unmatched
            # Need to add: matched to wrong track
            for tid, count in trk_counts.items():
                if tid != best_trk_id:
                    idfn_total += count
                    idfp_total += count

    # Compute metrics
    mota = 1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)
    motp = total_iou / max(total_matches, 1)
    
    # IDF1
    idf1 = (2 * idtp_total) / max(2 * idtp_total + idfp_total + idfn_total, 1)

    # Mostly tracked / mostly lost
    gt_track_stats = defaultdict(lambda: {'total': 0, 'tracked': 0})
    for f in sorted(gt_per_frame.keys()):
        gt_items = gt_per_frame[f]
        trk_items = traj_per_frame.get(f, [])

        for gt_bbox, gt_id in gt_items:
            gt_track_stats[gt_id]['total'] += 1

        if len(gt_items) > 0 and len(trk_items) > 0:
            gt_bboxes = [g[0] for g in gt_items]
            gt_ids_list = [g[1] for g in gt_items]
            trk_bboxes = [t[1] for t in trk_items]
            trk_ids_list = [t[0] for t in trk_items]

            iou_mat = compute_iou_matrix(gt_bboxes, trk_bboxes)
            cost_mat = 1.0 - iou_mat
            matched, _, _ = hungarian_matching(cost_mat, threshold=1.0 - iou_threshold)

            for gt_idx, trk_idx in matched:
                gt_id = gt_ids_list[gt_idx]
                gt_track_stats[gt_id]['tracked'] += 1

    mostly_tracked = sum(1 for s in gt_track_stats.values()
                         if s['tracked'] >= 0.8 * s['total'])
    mostly_lost = sum(1 for s in gt_track_stats.values()
                      if s['tracked'] < 0.2 * s['total'])
    partially_tracked = len(gt_track_stats) - mostly_tracked - mostly_lost

    return {
        'MOTA': round(mota, 4),
        'MOTP': round(motp, 4),
        'IDF1': round(idf1, 4),
        'ID_Switches': total_idsw,
        'FP': total_fp,
        'FN': total_fn,
        'TP': total_tp,
        'Total_GT': total_gt,
        'Mostly_Tracked': mostly_tracked,
        'Mostly_Lost': mostly_lost,
        'Partially_Tracked': partially_tracked
    }


def run_tracker(tracker_class, tracker_kwargs, sequence_data):
    """
    Run a tracker on a sequence and return trajectories.
    
    Returns: dict mapping track_id -> list of (frame, bbox)
    """
    tracker = tracker_class(**tracker_kwargs)
    trajectories = defaultdict(list)

    for frame_data in sequence_data:
        f = frame_data['frame']
        detections = frame_data['detections']

        results = tracker.update(detections)

        for track_id, bbox in results:
            trajectories[track_id].append((f, bbox))

    return dict(trajectories)
