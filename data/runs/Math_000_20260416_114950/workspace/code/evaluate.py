import json
import numpy as np
from tracker import iou

def evaluate_tracking(gt_data_path, pred_data_path):
    with open(gt_data_path, 'r') as f:
        gt_data = json.load(f)
        
    with open(pred_data_path, 'r') as f:
        pred_data = json.load(f)
        
    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    
    # Track ID mapping: GT ID -> Pred ID (for MOTA ID Sw)
    id_mapping = {}
    
    # To correctly calculate MOTA, we need to track the mapping from previous frames
    # and only count ID switches when a tracked object changes its assigned ID.
    
    for frame_idx in range(len(gt_data)):
        gt_frame = gt_data[frame_idx]
        pred_frame = pred_data[frame_idx]
        
        gt_bboxes = gt_frame['gt_bboxes']
        gt_ids = gt_frame['gt_ids']
        
        pred_tracks = pred_frame['tracks']
        pred_bboxes = [t['bbox'] for t in pred_tracks]
        pred_ids = [t['track_id'] for t in pred_tracks]
        
        total_gt += len(gt_bboxes)
        total_pred += len(pred_bboxes)
        
        if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
            continue
            
        if len(gt_bboxes) == 0:
            false_positives += len(pred_bboxes)
            continue
            
        if len(pred_bboxes) == 0:
            false_negatives += len(gt_bboxes)
            continue
            
        # Match GT and Pred
        iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)))
        for i, gt_box in enumerate(gt_bboxes):
            for j, pred_box in enumerate(pred_bboxes):
                iou_matrix[i, j] = iou(gt_box, pred_box)
                
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        # Convert to cost matrix
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_gt = []
        matched_pred = []
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= 0.5:
                matched_gt.append(r)
                matched_pred.append(c)
                true_positives += 1
                
                gt_id = gt_ids[r]
                pred_id = pred_ids[c]
                
                if gt_id in id_mapping:
                    if id_mapping[gt_id] != pred_id:
                        id_switches += 1
                        id_mapping[gt_id] = pred_id
                else:
                    id_mapping[gt_id] = pred_id
                    
        false_negatives += len(gt_bboxes) - len(matched_gt)
        false_positives += len(pred_bboxes) - len(matched_pred)
        
    mota = 1 - (false_negatives + false_positives + id_switches) / total_gt if total_gt > 0 else 0
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'MOTA': mota,
        'IDF1': f1, # Approximation for now
        'ID_Switches': id_switches,
        'FP': false_positives,
        'FN': false_negatives,
        'TP': true_positives,
        'Total_GT': total_gt,
        'Total_Pred': total_pred
    }

print("ByteTrack Evaluation:")
print(evaluate_tracking('data/simulated_sequence.json', 'outputs/bytetrack_results.json'))

print("\nSparseTrack Evaluation:")
print(evaluate_tracking('data/simulated_sequence.json', 'outputs/sparsetrack_results.json'))
