"""
sparsetrack.py
================
Faithful re-implementation of the SparseTrack association strategy
(Liu et al., 2023 — "SparseTrack: Multi-Object Tracking by Performing
Scene Decomposition based on Pseudo-Depth").

Novelty over ByteTrack:
  * Pseudo-depth d_i = bbox bottom y-coordinate (larger d -> closer to camera).
    The intuition is that for a fixed camera, objects near the lower image
    border are closer; sorting by bottom-y gives a depth-like ordering.
  * Depth Cascade Matching (DCM): the dense scene is decomposed into
    sparse subsets by depth level. Within each level we run IoU-Hungarian
    locally; unmatched tracks/detections cascade to subsequent levels.
  * The high/low score split (ByteTrack) is preserved on top of DCM.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
from tracker_core import STrack, iou_matrix


def _linear_assign(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, mr, mc = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            mr.add(r); mc.add(c)
    ur = [r for r in range(cost.shape[0]) if r not in mr]
    uc = [c for c in range(cost.shape[1]) if c not in mc]
    return matches, ur, uc


def _pseudo_depth(boxes: np.ndarray) -> np.ndarray:
    """boxes: (N,4) xyxy.  Returns the bottom-y per box."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(boxes, dtype=np.float64)[:, 3]


def _depth_cascade_match(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    n_levels: int,
    iou_cost_thresh: float,
):
    """
    Sort tracks and detections into K depth bins by bottom-y, then run
    Hungarian matching within each bin (from deepest = smallest bottom-y to
    shallowest = largest bottom-y, which matches the SparseTrack near->far
    cascade after argsort-flip). Unmatched leftovers fall through to a final
    global pass.

    Returns (matches_global_idx, unmatched_track_idx, unmatched_det_idx)
    where indices are into the full (un-sorted) arrays.
    """
    n_t = len(track_boxes)
    n_d = len(det_boxes)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    t_depth = _pseudo_depth(track_boxes)
    d_depth = _pseudo_depth(det_boxes)

    # Build bin edges from the union of depths (so each bin is non-empty in
    # at least the global sense). Use quantile edges over union of depths.
    union = np.concatenate([t_depth, d_depth])
    qs = np.linspace(0, 1, n_levels + 1)
    edges = np.quantile(union, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    matched_global: List[Tuple[int, int]] = []
    used_t = np.zeros(n_t, dtype=bool)
    used_d = np.zeros(n_d, dtype=bool)

    # Process from far -> near (low bottom-y to high) to mimic the paper's
    # cascade order; equivalent up to symmetry of the sparsification.
    for lvl in range(n_levels):
        lo, hi = edges[lvl], edges[lvl + 1]
        t_in = np.where((~used_t) & (t_depth > lo) & (t_depth <= hi))[0]
        d_in = np.where((~used_d) & (d_depth > lo) & (d_depth <= hi))[0]
        if len(t_in) == 0 or len(d_in) == 0:
            continue
        sub_iou = iou_matrix(track_boxes[t_in], det_boxes[d_in])
        cost = 1.0 - sub_iou
        matches, _ur, _uc = _linear_assign(cost, iou_cost_thresh)
        for ri, ci in matches:
            gi, gj = int(t_in[ri]), int(d_in[ci])
            matched_global.append((gi, gj))
            used_t[gi] = True
            used_d[gj] = True

    # Final global pass on whatever is left (cross-bin recovery).
    rem_t = np.where(~used_t)[0]
    rem_d = np.where(~used_d)[0]
    if len(rem_t) > 0 and len(rem_d) > 0:
        sub_iou = iou_matrix(track_boxes[rem_t], det_boxes[rem_d])
        cost = 1.0 - sub_iou
        matches, _ur, _uc = _linear_assign(cost, iou_cost_thresh)
        for ri, ci in matches:
            gi, gj = int(rem_t[ri]), int(rem_d[ci])
            matched_global.append((gi, gj))
            used_t[gi] = True
            used_d[gj] = True

    unmatched_t = list(np.where(~used_t)[0])
    unmatched_d = list(np.where(~used_d)[0])
    return matched_global, unmatched_t, unmatched_d


class SparseTrackTracker:
    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        match_thresh: float = 0.8,
        match_thresh_low: float = 0.5,
        match_thresh_unconf: float = 0.7,
        max_time_lost: int = 30,
        n_levels: int = 4,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_low = match_thresh_low
        self.match_thresh_unconf = match_thresh_unconf
        self.max_time_lost = max_time_lost
        self.n_levels = n_levels

        self.tracked: List[STrack] = []
        self.lost: List[STrack] = []
        self.removed: List[STrack] = []
        self.frame_id = -1
        STrack.reset_id()

    def update(self, dets: np.ndarray, scores: np.ndarray) -> List[STrack]:
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)
        dets_high, sc_high = dets[high_mask], scores[high_mask]
        dets_low, sc_low = dets[low_mask], scores[low_mask]

        unconfirmed = [t for t in self.tracked if t.state == "tentative"]
        confirmed = [t for t in self.tracked if t.state != "tentative"]
        track_pool = confirmed + self.lost

        for t in track_pool:
            t.predict()
        for t in unconfirmed:
            t.predict()

        # ----- Stage 1: DCM on high-score dets vs (confirmed+lost) -----
        track_boxes = np.array([t.bbox for t in track_pool]) if track_pool else np.zeros((0, 4))
        matches, u_track_idx, u_det_idx = _depth_cascade_match(
            track_boxes, dets_high, n_levels=self.n_levels, iou_cost_thresh=self.match_thresh
        )

        activated, lost_now = [], []
        for ti, di in matches:
            t = track_pool[ti]
            t.update(dets_high[di], sc_high[di], self.frame_id)
            if t in self.lost:
                self.lost.remove(t)
            activated.append(t)

        unmatched_tracks_s1 = [track_pool[i] for i in u_track_idx]
        unmatched_dets_high = [dets_high[i] for i in u_det_idx]
        unmatched_sc_high = [sc_high[i] for i in u_det_idx]

        # ----- Stage 2: DCM on low-score dets vs unmatched-tracked -----
        s2_pool = [t for t in unmatched_tracks_s1 if t.state != "lost"]
        s2_lost = [t for t in unmatched_tracks_s1 if t.state == "lost"]
        s2_boxes = np.array([t.bbox for t in s2_pool]) if s2_pool else np.zeros((0, 4))
        m2, u_t2, _ = _depth_cascade_match(
            s2_boxes, dets_low, n_levels=self.n_levels, iou_cost_thresh=self.match_thresh_low
        )
        for ti, di in m2:
            t = s2_pool[ti]
            t.update(dets_low[di], sc_low[di], self.frame_id)
            activated.append(t)
        for ti in u_t2:
            t = s2_pool[ti]
            if t.state != "lost":
                t.mark_lost()
                lost_now.append(t)
        for t in s2_lost:
            lost_now.append(t)

        # ----- Stage 3: unconfirmed vs leftover high-score dets (single-pass IoU) -----
        if unmatched_dets_high:
            unconf_boxes = np.array([t.bbox for t in unconfirmed]) if unconfirmed else np.zeros((0, 4))
            d_arr = np.array(unmatched_dets_high)
            ious3 = iou_matrix(unconf_boxes, d_arr)
            cost3 = 1.0 - ious3
            m3, u_unc, u_det_h = _linear_assign(cost3, self.match_thresh_unconf)
            for ti, di in m3:
                t = unconfirmed[ti]
                t.update(d_arr[di], unmatched_sc_high[di], self.frame_id)
                activated.append(t)
            for ti in u_unc:
                t = unconfirmed[ti]
                t.mark_removed()
                self.removed.append(t)
            leftover_dets = [d_arr[i] for i in u_det_h]
            leftover_sc = [unmatched_sc_high[i] for i in u_det_h]
        else:
            for t in unconfirmed:
                t.mark_removed()
                self.removed.append(t)
            leftover_dets, leftover_sc = [], []

        # ----- Init new tracks -----
        for d, s in zip(leftover_dets, leftover_sc):
            if s >= self.new_track_thresh:
                nt = STrack(np.asarray(d), s, self.frame_id)
                activated.append(nt)

        # ----- Lost track aging -----
        new_lost = []
        for t in self.lost + lost_now:
            if t.state == "removed":
                continue
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.mark_removed()
                self.removed.append(t)
            else:
                if t not in new_lost:
                    new_lost.append(t)
        self.lost = new_lost

        self.tracked = [t for t in activated if t.state not in ("lost", "removed")]
        return [t for t in self.tracked if t.frame_id == self.frame_id and t.state in ("tracked", "tentative")]
