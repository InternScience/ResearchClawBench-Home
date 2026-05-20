"""Few-shot learning and multi-scale fusion module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestCentroid

class PrototypeNetwork(nn.Module):
    """Prototypical network for few-shot attack detection."""
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    def compute_prototypes(self, support_feats, support_labels, n_classes):
        prototypes = []
        for c in range(n_classes):
            mask = support_labels == c
            if mask.sum() > 0:
                proto = support_feats[mask].mean(dim=0)
            else:
                proto = torch.zeros(support_feats.shape[1], device=support_feats.device)
            prototypes.append(proto)
        return torch.stack(prototypes)
    
    def predict(self, query_feats, prototypes):
        dists = torch.cdist(query_feats, prototypes)
        return dists.argmin(dim=-1)


class BiSimilarityFusion(nn.Module):
    """Bi-similarity multi-scale fusion for few-shot detection."""
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.scale_net1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        self.scale_net2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        self.fusion = nn.Linear(128, 64)
    
    def forward(self, x):
        z1 = self.scale_net1(x)
        z2 = self.scale_net2(x)
        z_fused = self.fusion(torch.cat([z1, z2], dim=-1))
        return z_fused


def create_few_shot_episodes(features, labels, n_way=5, n_shot=5, n_query=15, n_episodes=100):
    """Create few-shot episodes from data."""
    unique_classes = np.unique(labels)
    episodes = []
    
    for _ in range(n_episodes):
        selected_classes = np.random.choice(unique_classes, size=min(n_way, len(unique_classes)), replace=False)
        
        support_feats, support_labels = [], []
        query_feats, query_labels = [], []
        
        for cls in selected_classes:
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) < n_shot + n_query:
                continue
            selected = np.random.choice(cls_indices, size=n_shot + n_query, replace=False)
            
            for j in range(n_shot):
                support_feats.append(features[selected[j]])
                support_labels.append(cls)
            for j in range(n_shot, n_shot + n_query):
                query_feats.append(features[selected[j]])
                query_labels.append(cls)
        
        if len(support_feats) > 0:
            episodes.append({
                'support_feats': torch.tensor(np.array(support_feats), dtype=torch.float32),
                'support_labels': torch.tensor(support_labels, dtype=torch.long),
                'query_feats': torch.tensor(np.array(query_feats), dtype=torch.float32),
                'query_labels': torch.tensor(query_labels, dtype=torch.long),
                'n_classes': len(selected_classes)
            })
    
    return episodes


def evaluate_open_set(features, labels, known_classes, unknown_classes):
    """Evaluate unknown attack detection: train on known, test on known+unknown."""
    known_mask = np.isin(labels, known_classes)
    unknown_mask = np.isin(labels, unknown_classes)
    
    train_feats = features[known_mask]
    train_labels = labels[known_mask]
    
    test_feats = features[known_mask | unknown_mask]
    test_labels = labels[known_mask | unknown_mask]
    
    clf = NearestCentroid()
    clf.fit(train_feats, train_labels)
    
    preds = clf.predict(test_feats)
    
    # For unknown detection: threshold-based rejection
    centroids = clf.centroids_
    dists = np.min([np.linalg.norm(test_feats - c, axis=1) for c in centroids], axis=0)
    threshold = np.percentile(dists[known_mask], 95)
    unknown_pred = (dists > threshold).astype(int)
    
    # Binary: known vs unknown
    binary_true = np.isin(test_labels, unknown_classes).astype(int)
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    return {
        'accuracy': accuracy_score(binary_true, unknown_pred),
        'f1': f1_score(binary_true, unknown_pred, average='binary', zero_division=0),
        'precision': precision_score(binary_true, unknown_pred, average='binary', zero_division=0),
        'recall': recall_score(binary_true, unknown_pred, average='binary', zero_division=0)
    }
