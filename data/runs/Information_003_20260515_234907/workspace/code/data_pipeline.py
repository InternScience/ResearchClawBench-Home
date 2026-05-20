"""Data loading, preprocessing, and statistical disentanglement pipeline."""
import torch
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

def load_data(path="data/NF-UNSW-NB15-v2_3d.pt"):
    """Load temporal graph data."""
    data = torch.load(path, weights_only=False)
    return data

def extract_flat_features(data):
    """Extract flat feature matrix and labels."""
    features = data.msg.numpy()  # [N, 40]
    labels = data.label.numpy()  # [N] binary
    attacks = data.attack.numpy()  # [N] multi-class
    timestamps = data.t.numpy()
    src = data.src.numpy()
    dst = data.dst.numpy()
    return features, labels, attacks, timestamps, src, dst

def statistical_disentanglement(features, n_components=20, mi_threshold=0.01):
    """
    Statistical disentanglement: combine PCA, ICA, and MI-based feature selection.
    
    Steps:
    1. Standardize features
    2. Apply PCA for initial dimensionality reduction
    3. Apply ICA to find independent components
    4. Use mutual information to select most discriminative components
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # PCA for decorrelation
    pca = PCA(n_components=min(n_components, features.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # ICA for statistical independence (disentanglement)
    ica = FastICA(n_components=min(n_components, X_pca.shape[1]), 
                  max_iter=1000, random_state=42, tol=0.001)
    X_ica = ica.fit_transform(X_pca)
    
    return X_ica, scaler, pca, ica

def compute_feature_importance(features, labels):
    """Compute mutual information between each feature and labels."""
    mi_scores = mutual_info_classif(features, labels, random_state=42)
    return mi_scores
