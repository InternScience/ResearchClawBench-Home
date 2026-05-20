"""
Feature engineering from raw pose coordinates for SimBA-style behavior classification.
Input: raw pose coordinates for two mice (8 keypoints × 3 coordinates × 2 mice)
Output: engineered feature matrix for classifier training.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy import stats

def load_raw_data(features_path, targets_path):
    """Load raw pose features and behavior labels."""
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    
    # Drop the unnamed index column
    if 'Unnamed: 0' in features.columns:
        features = features.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in targets.columns:
        targets = targets.drop(columns=['Unnamed: 0'])
    
    return features, targets

def compute_bodypart_arrays(features_df):
    """Extract body part coordinate arrays for both mice."""
    bp_names = ['Nose', 'Ear_left', 'Ear_right', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']
    coords = ['x', 'y', 'p']
    
    mouse1 = {}
    mouse2 = {}
    
    for bp in bp_names:
        m1 = {}
        m2 = {}
        for c in coords:
            col1 = f'{bp}_1_{c}'
            col2 = f'{bp}_2_{c}'
            if col1 in features_df.columns:
                m1[c] = features_df[col1].values
            if col2 in features_df.columns:
                m2[c] = features_df[col2].values
        mouse1[bp] = m1
        mouse2[bp] = m2
    
    return mouse1, mouse2, bp_names

def euclidean(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def compute_feature_matrix(features_df):
    """Compute engineered feature matrix from raw pose data."""
    n_frames = len(features_df)
    mouse1, mouse2, bp_names = compute_bodypart_arrays(features_df)
    
    feature_dict = {}
    
    # Helper to get coords
    def get_xy(mouse, bp):
        return mouse[bp]['x'], mouse[bp]['y']
    
    # === Within-mouse features ===
    for m_idx, mouse in enumerate([mouse1, mouse2]):
        m_tag = f'M{m_idx+1}'
        
        # Nose to tail distance (body length proxy)
        nx, ny = get_xy(mouse, 'Nose')
        tx, ty = get_xy(mouse, 'Tail_base')
        feature_dict[f'{m_tag}_nose_to_tail'] = euclidean(nx, ny, tx, ty)
        
        # Width (lateral left to lateral right)
        llx, lly = get_xy(mouse, 'Lat_left')
        lrx, lry = get_xy(mouse, 'Lat_right')
        feature_dict[f'{m_tag}_width'] = euclidean(llx, lly, lrx, lry)
        
        # Ear distance
        elx, ely = get_xy(mouse, 'Ear_left')
        erx, ery = get_xy(mouse, 'Ear_right')
        feature_dict[f'{m_tag}_ear_distance'] = euclidean(elx, ely, erx, ery)
        
        # Nose to centroid
        cx, cy = get_xy(mouse, 'Center')
        feature_dict[f'{m_tag}_nose_to_centroid'] = euclidean(nx, ny, cx, cy)
        
        # Nose to lateral left/right
        feature_dict[f'{m_tag}_nose_to_lat_left'] = euclidean(nx, ny, llx, lly)
        feature_dict[f'{m_tag}_nose_to_lat_right'] = euclidean(nx, ny, lrx, lry)
        
        # Centroid to lateral left/right
        feature_dict[f'{m_tag}_centroid_to_lat_left'] = euclidean(cx, cy, llx, lly)
        feature_dict[f'{m_tag}_centroid_to_lat_right'] = euclidean(cx, cy, lrx, lry)
        
        # Polygon area (approximate using convex hull of body parts)
        body_parts_xy = []
        for bp in bp_names:
            if bp != 'Tail_end':  # Tail end can be noisy
                bx, by = get_xy(mouse, bp)
                body_parts_xy.append(np.column_stack([bx, by]))
        
        # Compute convex hull area per frame
        hull_areas = np.zeros(n_frames)
        for i in range(n_frames):
            pts = np.array([[bp[i, 0], bp[i, 1]] for bp in body_parts_xy])
            try:
                hull = ConvexHull(pts)
                hull_areas[i] = hull.volume  # area in 2D
            except:
                hull_areas[i] = 0
        feature_dict[f'{m_tag}_poly_area'] = hull_areas
    
    # === Between-mouse features ===
    m1_cx, m1_cy = get_xy(mouse1, 'Center')
    m2_cx, m2_cy = get_xy(mouse2, 'Center')
    feature_dict['Centroid_distance'] = euclidean(m1_cx, m1_cy, m2_cx, m2_cy)
    
    m1_nx, m1_ny = get_xy(mouse1, 'Nose')
    m2_nx, m2_ny = get_xy(mouse2, 'Nose')
    feature_dict['Nose_to_nose_distance'] = euclidean(m1_nx, m1_ny, m2_nx, m2_ny)
    
    # Cross mouse distances
    m2_llx, m2_lly = get_xy(mouse2, 'Lat_left')
    m2_lrx, m2_lry = get_xy(mouse2, 'Lat_right')
    m1_llx, m1_lly = get_xy(mouse1, 'Lat_left')
    m1_lrx, m1_lry = get_xy(mouse1, 'Lat_right')
    
    feature_dict['M1_Nose_to_M2_lat_left'] = euclidean(m1_nx, m1_ny, m2_llx, m2_lly)
    feature_dict['M1_Nose_to_M2_lat_right'] = euclidean(m1_nx, m1_ny, m2_lrx, m2_lry)
    feature_dict['M2_Nose_to_M1_lat_left'] = euclidean(m2_nx, m2_ny, m1_llx, m1_lly)
    feature_dict['M2_Nose_to_M1_lat_right'] = euclidean(m2_nx, m2_ny, m1_lrx, m1_lry)
    
    m1_tbx, m1_tby = get_xy(mouse1, 'Tail_base')
    m2_tbx, m2_tby = get_xy(mouse2, 'Tail_base')
    feature_dict['M1_Nose_to_M2_tail_base'] = euclidean(m1_nx, m1_ny, m2_tbx, m2_tby)
    feature_dict['M2_Nose_to_M1_tail_base'] = euclidean(m2_nx, m2_ny, m1_tbx, m1_tby)
    
    # === Movement features (frame-to-frame displacements) ===
    movement_bps = ['Nose', 'Center', 'Tail_base', 'Tail_end', 'Ear_left', 'Ear_right', 'Lat_left', 'Lat_right']
    
    for m_idx, mouse in enumerate([mouse1, mouse2]):
        m_tag = f'M{m_idx+1}'
        for bp in movement_bps:
            bx, by = get_xy(mouse, bp)
            dx = np.diff(bx, prepend=bx[0])
            dy = np.diff(by, prepend=by[0])
            movement = np.sqrt(dx**2 + dy**2)
            feature_dict[f'Movement_{m_tag}_{bp.lower()}'] = movement
    
    # === Rolling window statistics ===
    window_sizes = [2, 5, 6, 7, 10, 15]
    
    # Centroid distance rolling stats
    cd = feature_dict['Centroid_distance']
    nn = feature_dict['Nose_to_nose_distance']
    
    for w in window_sizes:
        if w <= n_frames:
            # Rolling mean
            cd_rolling = pd.Series(cd).rolling(window=w, min_periods=1).mean().values
            nn_rolling = pd.Series(nn).rolling(window=w, min_periods=1).mean().values
            feature_dict[f'Centroid_distance_mean_{w}'] = cd_rolling
            feature_dict[f'Nose_to_nose_distance_mean_{w}'] = nn_rolling
            
            # Rolling std
            cd_std = pd.Series(cd).rolling(window=w, min_periods=1).std().fillna(0).values
            nn_std = pd.Series(nn).rolling(window=w, min_periods=1).std().fillna(0).values
            feature_dict[f'Centroid_distance_std_{w}'] = cd_std
            feature_dict[f'Nose_to_nose_distance_std_{w}'] = nn_std
    
    # Total movement per mouse (sum of all body part movements)
    for m_idx, mouse in enumerate([mouse1, mouse2]):
        m_tag = f'M{m_idx+1}'
        total_movement = np.zeros(n_frames)
        for bp in movement_bps:
            total_movement += feature_dict[f'Movement_{m_tag}_{bp.lower()}']
        feature_dict[f'Total_movement_{m_tag}'] = total_movement
    
    feature_dict['Total_movement_both'] = feature_dict['Total_movement_M1'] + feature_dict['Total_movement_M2']
    
    # Rolling window stats for total movement
    for w in window_sizes:
        if w <= n_frames:
            tm = feature_dict['Total_movement_both']
            feature_dict[f'Total_movement_both_mean_{w}'] = pd.Series(tm).rolling(window=w, min_periods=1).mean().values
            feature_dict[f'Total_movement_both_std_{w}'] = pd.Series(tm).rolling(window=w, min_periods=1).std().fillna(0).values
    
    # === Angle features ===
    # Mouse body angle (nose to tail base vector)
    for m_idx, mouse in enumerate([mouse1, mouse2]):
        m_tag = f'M{m_idx+1}'
        nx, ny = get_xy(mouse, 'Nose')
        tx, ty = get_xy(mouse, 'Tail_base')
        angle = np.arctan2(ty - ny, tx - nx)
        feature_dict[f'{m_tag}_angle'] = angle
    
    # Angle between mice (angle between their body axes)
    m1_angle = feature_dict['M1_angle']
    m2_angle = feature_dict['M2_angle']
    feature_dict['Angle_between_mice'] = np.abs(m1_angle - m2_angle)
    # Wrap to [0, pi]
    feature_dict['Angle_between_mice'] = np.minimum(feature_dict['Angle_between_mice'], 
                                                       2*np.pi - feature_dict['Angle_between_mice'])
    
    # === Probability-related features ===
    # Low probability detections
    for m_idx, mouse in enumerate([mouse1, mouse2]):
        m_tag = f'M{m_idx+1}'
        low_prob_count = np.zeros(n_frames)
        for bp in bp_names:
            p_vals = mouse[bp]['p']
            low_prob_count += (p_vals < 0.1).astype(float)
        feature_dict[f'{m_tag}_low_prob_detections'] = low_prob_count
    
    # === Polygon size change ===
    for m_idx in [1, 2]:
        m_tag = f'M{m_idx}'
        pa = feature_dict[f'{m_tag}_poly_area']
        feature_dict[f'{m_tag}_polygon_size_change'] = np.diff(pa, prepend=pa[0])
    
    # === Additional derived features ===
    # Ratio features
    eps = 1e-8
    feature_dict['Width_ratio'] = feature_dict['M1_width'] / (feature_dict['M2_width'] + eps)
    feature_dict['Nose_to_tail_ratio'] = feature_dict['M1_nose_to_tail'] / (feature_dict['M2_nose_to_tail'] + eps)
    
    # Build the final DataFrame
    feature_df = pd.DataFrame(feature_dict)
    
    # Drop NaN and inf
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)
    
    return feature_df

def main():
    features_path = 'data/Together_1_features_extracted.csv'
    targets_path = 'data/Together_1_targets_inserted.csv'
    
    print("Loading raw data...")
    features, targets = load_raw_data(features_path, targets_path)
    print(f"Raw features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    print("\nEngineering features...")
    X = compute_feature_matrix(features)
    print(f"Engineered features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Save engineered features
    print("\nSaving engineered features...")
    X.to_csv('outputs/engineered_features.csv', index=False)
    
    # Save labels alongside
    y = targets[['Attack', 'Sniffing']].copy()
    y.to_csv('outputs/behavior_labels.csv', index=False)
    
    print("Done. Files saved to outputs/")
    
    return X, y

if __name__ == '__main__':
    X, y = main()
