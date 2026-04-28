"""
bytetrack.py
==============
Minimal, faithful re-implementation of ByteTrack association logic
(Zhang et al., ECCV 2022) on top of a SORT-style Kalman filter.

Key idea:
  Stage-1: high-score detections -> match to active+lost tracks via IoU
  Stage-2: low-score detections  -> match to remaining unmatched tracks
  Unmatched high-score detections -> new tracks (after 1 confirmation frame)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
from tracker_core import STrack, iou_matrix


def _linear_assign(cost: np.ndarray, thresh: float):
    """Hungarian assignment with cost-thresh gate. cost has high values for bad matches.
    Returns matches (i,j), unmatched rows, unmatched cols.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, ur, uc = [], [], []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    ur = [r for r in range(cost.shape[0]) if r not in matched_rows]
    uc = [c for c in range(cost.shape[1]) if c not in matched_cols]
    return matches, ur, uc


class ByteTrackTracker:
    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        match_thresh: float = 0.8,        # IoU >= 1 - 0.8 = 0.2
        match_thresh_low: float = 0.5,    # IoU >= 0.5 for low-score 2nd assoc
        match_thresh_unconf: float = 0.7,
        max_time_lost: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_low = match_thresh_low
        self.match_thresh_unconf = match_thresh_unconf
        self.max_time_lost = max_time_lost

        self.tracked: List[STrack] = []
        self.lost: List[STrack] = []
        self.removed: List[STrack] = []
        self.frame_id = -1
        STrack.reset_id()

    # main update step
    def update(self, dets: np.ndarray, scores: np.ndarray) -> List[STrack]:
        """dets: (N,4) xyxy float, scores: (N,) float
        Returns list of currently 'tracked' STracks for this frame."""
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)
        dets_high, sc_high = dets[high_mask], scores[high_mask]
        dets_low, sc_low = dets[low_mask], scores[low_mask]

        # split tracks into confirmed/lost vs tentative
        unconfirmed = [t for t in self.tracked if t.state == "tentative"]
        confirmed = [t for t in self.tracked if t.state != "tentative"]
        track_pool = confirmed + self.lost

        # Predict all
        for t in track_pool:
            t.predict()
        for t in unconfirmed:
            t.predict()

        # ----- Stage 1: high-score association with confirmed+lost -----
        track_boxes = np.array([t.bbox for t in track_pool]) if track_pool else np.zeros((0, 4))
        ious = iou_matrix(track_boxes, dets_high)
        cost = 1.0 - ious
        matches, u_track_idx, u_det_idx = _linear_assign(cost, self.match_thresh)

        activated, refind, lost_now = [], [], []
        for ti, di in matches:
            t = track_pool[ti]
            t.update(dets_high[di], sc_high[di], self.frame_id)
            if t in self.lost:
                self.lost.remove(t)
                refind.append(t)
            else:
                activated.append(t)

        unmatched_tracks_stage1 = [track_pool[i] for i in u_track_idx]
        unmatched_dets_high = [dets_high[i] for i in u_det_idx]
        unmatched_sc_high = [sc_high[i] for i in u_det_idx]

        # ----- Stage 2: low-score association with the unmatched (tracked-only) -----
        s2_pool = [t for t in unmatched_tracks_stage1 if t.state != "lost"]
        s2_lost = [t for t in unmatched_tracks_stage1 if t.state == "lost"]
        track_boxes_s2 = np.array([t.bbox for t in s2_pool]) if s2_pool else np.zeros((0, 4))
        ious2 = iou_matrix(track_boxes_s2, dets_low)
        cost2 = 1.0 - ious2
        matches2, u_track2_idx, _ = _linear_assign(cost2, self.match_thresh_low)
        for ti, di in matches2:
            t = s2_pool[ti]
            t.update(dets_low[di], sc_low[di], self.frame_id)
            activated.append(t)

        # tracks left after stage2 -> mark lost
        for ti in u_track2_idx:
            t = s2_pool[ti]
            if t.state != "lost":
                t.mark_lost()
                lost_now.append(t)
        for t in s2_lost:
            # already lost, stay lost
            lost_now.append(t)

        # ----- Stage 3: unconfirmed (tentative) tracks vs unmatched high-score detections -----
        if unmatched_dets_high:
            unconf_boxes = np.array([t.bbox for t in unconfirmed]) if unconfirmed else np.zeros((0, 4))
            d_arr = np.array(unmatched_dets_high)
            ious3 = iou_matrix(unconf_boxes, d_arr)
            cost3 = 1.0 - ious3
            matches3, u_unconf_idx, u_det_h_idx = _linear_assign(cost3, self.match_thresh_unconf)
            for ti, di in matches3:
                t = unconfirmed[ti]
                t.update(d_arr[di], unmatched_sc_high[di], self.frame_id)
                activated.append(t)
            # leftover unconfirmed -> remove
            for ti in u_unconf_idx:
                t = unconfirmed[ti]
                t.mark_removed()
                self.removed.append(t)
            leftover_dets = [d_arr[i] for i in u_det_h_idx]
            leftover_sc = [unmatched_sc_high[i] for i in u_det_h_idx]
        else:
            for t in unconfirmed:
                t.mark_removed()
                self.removed.append(t)
            leftover_dets, leftover_sc = [], []

        # ----- Init new tracks from remaining high-score dets -----
        for d, s in zip(leftover_dets, leftover_sc):
            if s >= self.new_track_thresh:
                nt = STrack(np.asarray(d), s, self.frame_id)
                activated.append(nt)

        # ----- Bookkeeping: lost tracks aging -----
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

        # update tracked set: confirmed+activated+refound
        self.tracked = [t for t in activated if t.state != "lost" and t.state != "removed"]
        # also keep any confirmed track that was matched (they're in activated already)
        # Return only tracks that updated this frame
        return [t for t in self.tracked if t.frame_id == self.frame_id and t.state in ("tracked", "tentative")]
