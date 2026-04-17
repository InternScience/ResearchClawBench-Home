"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection System 
with Multi-scale Fusion Learning

Inspired by 3D-IDS (KDD 2023) with extensions for few-shot learning.

Components:
1. Statistical Disentanglement - Feature weighting via MI minimization
2. Representational Disentanglement - Orthogonality regularization
3. Dynamic Graph Diffusion - Perona-Malik style diffusion
4. Multi-scale Fusion Learning - For few-shot attack detection
5. Binary + Multi-class Classification
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict, Counter
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. Statistical Disentanglement Module
# ============================================================
class StatisticalDisentanglement(nn.Module):
    """
    Disentangles traffic features by learning a weight vector that 
    minimizes mutual information between feature elements.
    Approximates the SMT-based optimization from 3D-IDS.
    """
    def __init__(self, feat_dim, w_min=0.1, w_max=2.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.w_min = w_min
        self.w_max = w_max
        # Learnable weight vector (will be constrained)
        self.raw_weights = nn.Parameter(torch.ones(feat_dim))
        
    def forward(self, features):
        """
        features: [batch, feat_dim] normalized features
        Returns: disentangled features [batch, feat_dim]
        """
        # Constrain weights to [w_min, w_max]
        w = torch.sigmoid(self.raw_weights) * (self.w_max - self.w_min) + self.w_min
        # Sort weights to maintain order-preserving property
        w_sorted, _ = torch.sort(w)
        # Hadamard product for disentanglement
        h = features * w_sorted.unsqueeze(0)
        return h
    
    def disentangle_loss(self, features):
        """
        Maximize distance between weighted components to minimize MI.
        """
        w = torch.sigmoid(self.raw_weights) * (self.w_max - self.w_min) + self.w_min
        w_sorted, _ = torch.sort(w)
        wf = features * w_sorted.unsqueeze(0)
        # Maximize spread: max(w_N*f_N - w_1*f_1)
        spread = wf[:, -1] - wf[:, 0]
        # Minimize curvature: sum |2*w_i*f_i - w_{i-1}*f_{i-1} - w_{i+1}*f_{i+1}|
        if wf.shape[1] > 2:
            curvature = torch.abs(
                2 * wf[:, 1:-1] - wf[:, :-2] - wf[:, 2:]
            ).sum(dim=1)
        else:
            curvature = torch.zeros(features.shape[0], device=features.device)
        # Loss: minimize negative spread + curvature
        loss = (-spread + curvature).mean()
        return loss


# ============================================================
# 2. Memory Module (TGN-style)
# ============================================================
class MemoryModule(nn.Module):
    """
    Maintains node memory using GRU updates, similar to TGN.
    """
    def __init__(self, num_nodes, memory_dim, msg_dim, time_dim=16):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.msg_dim = msg_dim
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Message function (RNN)
        self.msg_fn = nn.Linear(2 * memory_dim + msg_dim + time_dim + 2, memory_dim)
        
        # Memory updater (GRU)
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        
        # Memory storage (not a parameter, updated in-place)
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(num_nodes))
        
    def reset_memory(self):
        self.memory.zero_()
        self.last_update.zero_()
        
    def get_memory(self, node_ids):
        return self.memory[node_ids]
    
    def compute_message(self, src, dst, edge_feat, t, dt, src_layer, dst_layer):
        """Compute updating messages for source and destination nodes."""
        src_mem = self.memory[src]
        dst_mem = self.memory[dst]
        time_feat = self.time_encoder(dt.unsqueeze(-1))
        
        # Concatenate: src_mem, dst_mem, edge_feat, time_feat, layers
        msg_input = torch.cat([
            src_mem, dst_mem, edge_feat, time_feat,
            src_layer.float().unsqueeze(-1),
            dst_layer.float().unsqueeze(-1)
        ], dim=-1)
        
        msg_src = torch.relu(self.msg_fn(msg_input))
        
        # For dst, swap src and dst memory
        msg_input_dst = torch.cat([
            dst_mem, src_mem, edge_feat, time_feat,
            dst_layer.float().unsqueeze(-1),
            src_layer.float().unsqueeze(-1)
        ], dim=-1)
        msg_dst = torch.relu(self.msg_fn(msg_input_dst))
        
        return msg_src, msg_dst
    
    def update_memory(self, node_ids, messages):
        """Update memory for given nodes."""
        unique_nodes, inv = torch.unique(node_ids, return_inverse=True)
        # Aggregate messages for same node (mean)
        agg_msg = torch.zeros(unique_nodes.shape[0], self.memory_dim, 
                             device=messages.device)
        count = torch.zeros(unique_nodes.shape[0], 1, device=messages.device)
        agg_msg.scatter_add_(0, inv.unsqueeze(-1).expand_as(messages), messages)
        count.scatter_add_(0, inv.unsqueeze(-1), torch.ones_like(inv.unsqueeze(-1).float()))
        agg_msg = agg_msg / count.clamp(min=1)
        
        # GRU update
        old_mem = self.memory[unique_nodes]
        new_mem = self.gru(agg_msg, old_mem)
        self.memory[unique_nodes] = new_mem.detach()


class TimeEncoder(nn.Module):
    """Encode time differences using learnable Fourier features."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        
    def forward(self, t):
        return torch.cos(self.w(t))


# ============================================================
# 3. Representational Disentanglement
# ============================================================
class RepresentationalDisentanglement(nn.Module):
    """
    Ensures node representations maintain disentangled property
    via orthogonality regularization.
    """
    def __init__(self, rep_dim):
        super().__init__()
        self.rep_dim = rep_dim
        self.projection = nn.Linear(rep_dim, rep_dim)
        
    def forward(self, x):
        return self.projection(x)
    
    @staticmethod
    def disentangle_loss(x_t, x_t_prev):
        """
        L_dis = 0.5 * ||X(t) * X(t-1)^T - I||_F^2
        Encourages orthogonality between time-adjacent representations.
        """
        if x_t.shape[0] < 2:
            return torch.tensor(0.0, device=x_t.device)
        # Normalize
        x_t_norm = F.normalize(x_t, dim=-1)
        x_prev_norm = F.normalize(x_t_prev, dim=-1)
        # Compute correlation
        corr = torch.mm(x_t_norm, x_prev_norm.t())
        identity = torch.eye(min(corr.shape), device=corr.device)
        if corr.shape[0] != corr.shape[1]:
            # Use batch-wise disentanglement
            corr = torch.mm(x_t_norm.t(), x_prev_norm)
            identity = torch.eye(corr.shape[0], device=corr.device)
        loss = 0.5 * torch.norm(corr - identity, p='fro') ** 2
        return loss / max(x_t.shape[0], 1)


# ============================================================
# 4. Graph Diffusion Module
# ============================================================
class GraphDiffusion(nn.Module):
    """
    Multi-layer graph diffusion using Perona-Malik style nonlinear filtering.
    Simplified for CPU execution.
    """
    def __init__(self, node_dim, hidden_dim, num_steps=3):
        super().__init__()
        self.num_steps = num_steps
        self.K = nn.Linear(node_dim, hidden_dim)  # Transformation matrix
        
        # Layer-temporal coefficient
        self.layer_time_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Diffusivity function g(x) = exp(-|x|)
        self.sigma = lambda x: torch.exp(-torch.abs(x))
        
    def forward(self, x_src, x_dst, src_layer, dst_layer, dt):
        """
        Perform graph diffusion between connected nodes.
        x_src, x_dst: [batch, node_dim]
        Returns updated representations.
        """
        # Compute layer-temporal coefficients
        time_feat = dt.unsqueeze(-1)
        layer_feat = torch.stack([
            src_layer.float(), dst_layer.float()
        ], dim=-1)
        s_input = torch.cat([layer_feat, time_feat], dim=-1)
        s = torch.sigmoid(self.layer_time_mlp(s_input))  # [batch, 1]
        
        # Transform
        kx_src = self.K(x_src)
        kx_dst = self.K(x_dst)
        
        # Gradient on graph (difference between connected nodes)
        grad = kx_dst - kx_src  # [batch, hidden]
        
        # Perona-Malik diffusion: g(|grad|) * grad
        diffusion = self.sigma(grad) * grad * s
        
        # Euler integration for num_steps
        step_size = 1.0 / self.num_steps
        x_src_new = x_src.clone()
        x_dst_new = x_dst.clone()
        
        for _ in range(self.num_steps):
            x_src_new = x_src_new + step_size * diffusion
            x_dst_new = x_dst_new - step_size * diffusion
            
        return x_src_new, x_dst_new


# ============================================================
# 5. Multi-Scale Fusion Learning Module
# ============================================================
class MultiScaleFusion(nn.Module):
    """
    Fuses representations at multiple scales for better few-shot detection.
    Inspired by BSNet's bi-similarity approach.
    """
    def __init__(self, input_dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // s if input_dim // s > 0 else 1),
                nn.ReLU(),
                nn.Linear(input_dim // s if input_dim // s > 0 else 1, input_dim)
            ) for s in scales
        ])
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * len(scales), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x):
        """
        x: [batch, input_dim]
        Returns: fused multi-scale representation [batch, input_dim]
        """
        scale_reps = []
        for encoder in self.scale_encoders:
            scale_reps.append(encoder(x))
        
        # Concatenate and fuse
        concat = torch.cat(scale_reps, dim=-1)
        fused = self.fusion(concat)
        return fused


# ============================================================
# 6. DIDS-MFL Complete Model
# ============================================================
class DIDS_MFL(nn.Module):
    """
    Disentangled Dynamic Intrusion Detection System 
    with Multi-scale Fusion Learning
    """
    def __init__(self, num_nodes, feat_dim=40, memory_dim=32, 
                 hidden_dim=32, num_attacks=10, diffusion_steps=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        
        # Module 1: Statistical Disentanglement
        self.stat_disentangle = StatisticalDisentanglement(feat_dim)
        
        # Module 2: Memory (TGN-style)
        self.memory_module = MemoryModule(
            num_nodes, memory_dim, feat_dim, time_dim=16
        )
        
        # Module 3: Representational Disentanglement
        self.rep_disentangle = RepresentationalDisentanglement(memory_dim)
        
        # Module 4: Graph Diffusion
        self.graph_diffusion = GraphDiffusion(
            memory_dim, hidden_dim, num_steps=diffusion_steps
        )
        
        # Module 5: Multi-Scale Fusion
        self.multi_scale_fusion = MultiScaleFusion(
            memory_dim, scales=[1, 2, 4]
        )
        
        # Classifier heads
        # Binary classifier
        self.binary_classifier = nn.Sequential(
            nn.Linear(2 * memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
        
        # Multi-class classifier
        self.multi_classifier = nn.Sequential(
            nn.Linear(2 * memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_attacks)
        )
        
    def reset_memory(self):
        self.memory_module.reset_memory()
        
    def forward(self, src, dst, msg, t, dt, src_layer, dst_layer, 
                return_reps=False):
        """
        Forward pass for a batch of edges.
        """
        # Step 1: Statistical Disentanglement
        h = self.stat_disentangle(msg)
        
        # Step 2: Compute messages and update memory
        msg_src, msg_dst = self.memory_module.compute_message(
            src, dst, h, t, dt, src_layer, dst_layer
        )
        
        # Get previous representations
        x_src_prev = self.memory_module.get_memory(src).clone()
        x_dst_prev = self.memory_module.get_memory(dst).clone()
        
        # Update memory
        all_nodes = torch.cat([src, dst])
        all_msgs = torch.cat([msg_src, msg_dst])
        self.memory_module.update_memory(all_nodes, all_msgs)
        
        # Get updated representations
        x_src = self.memory_module.get_memory(src)
        x_dst = self.memory_module.get_memory(dst)
        
        # Step 3: Representational Disentanglement
        x_src = self.rep_disentangle(x_src)
        x_dst = self.rep_disentangle(x_dst)
        
        # Step 4: Graph Diffusion
        x_src_diff, x_dst_diff = self.graph_diffusion(
            x_src, x_dst, src_layer, dst_layer, dt
        )
        
        # Step 5: Multi-Scale Fusion
        x_src_fused = self.multi_scale_fusion(x_src_diff)
        x_dst_fused = self.multi_scale_fusion(x_dst_diff)
        
        # Edge representation
        edge_rep = torch.cat([x_src_fused, x_dst_fused], dim=-1)
        
        # Classification
        binary_logits = self.binary_classifier(edge_rep)
        multi_logits = self.multi_classifier(edge_rep)
        
        if return_reps:
            return binary_logits, multi_logits, edge_rep, x_src_prev, x_src
        
        return binary_logits, multi_logits, x_src_prev, x_src


# ============================================================
# 7. Baseline Models
# ============================================================
class MLPBaseline(nn.Module):
    """Simple MLP baseline for intrusion detection."""
    def __init__(self, feat_dim=40, hidden_dim=64, num_attacks=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.multi_head = nn.Linear(hidden_dim, num_attacks)
        
    def forward(self, msg):
        h = self.encoder(msg)
        return self.binary_head(h), self.multi_head(h)


class EGraphSAGEBaseline(nn.Module):
    """Simplified E-GraphSAGE baseline."""
    def __init__(self, feat_dim=40, hidden_dim=64, num_attacks=10):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        # Simple neighborhood aggregation
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.multi_head = nn.Linear(hidden_dim, num_attacks)
        
    def forward(self, msg, src_feat=None, dst_feat=None):
        h = self.edge_encoder(msg)
        if src_feat is not None and dst_feat is not None:
            h = self.agg(torch.cat([h, (src_feat + dst_feat) / 2], dim=-1))
        return self.binary_head(h), self.multi_head(h)


class TGNBaseline(nn.Module):
    """Simplified TGN baseline (without disentanglement)."""
    def __init__(self, num_nodes, feat_dim=40, memory_dim=32, 
                 hidden_dim=32, num_attacks=10):
        super().__init__()
        self.memory_module = MemoryModule(num_nodes, memory_dim, feat_dim)
        self.binary_classifier = nn.Sequential(
            nn.Linear(2 * memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
        self.multi_classifier = nn.Sequential(
            nn.Linear(2 * memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_attacks)
        )
        
    def reset_memory(self):
        self.memory_module.reset_memory()
        
    def forward(self, src, dst, msg, t, dt, src_layer, dst_layer):
        msg_src, msg_dst = self.memory_module.compute_message(
            src, dst, msg, t, dt, src_layer, dst_layer
        )
        x_src_prev = self.memory_module.get_memory(src).clone()
        all_nodes = torch.cat([src, dst])
        all_msgs = torch.cat([msg_src, msg_dst])
        self.memory_module.update_memory(all_nodes, all_msgs)
        x_src = self.memory_module.get_memory(src)
        x_dst = self.memory_module.get_memory(dst)
        edge_rep = torch.cat([x_src, x_dst], dim=-1)
        return self.binary_classifier(edge_rep), self.multi_classifier(edge_rep), x_src_prev, x_src


# ============================================================
# 8. Training and Evaluation Functions
# ============================================================
def compute_loss(binary_logits, multi_logits, labels, attacks,
                 x_prev, x_curr, stat_dis_module, msg,
                 alpha=0.1, beta=0.1, gamma=0.05):
    """
    Combined loss: L = L_int + alpha * L_smooth + beta * L_dis + gamma * L_stat
    """
    # Binary cross-entropy
    loss_binary = F.cross_entropy(binary_logits, labels)
    
    # Multi-class cross-entropy (only for attack samples)
    loss_multi = F.cross_entropy(multi_logits, attacks)
    
    # Smoothness loss
    loss_smooth = torch.norm(x_curr - x_prev, p=2, dim=-1).mean()
    
    # Representational disentanglement loss
    loss_dis = RepresentationalDisentanglement.disentangle_loss(x_curr, x_prev)
    
    # Statistical disentanglement loss
    loss_stat = stat_dis_module.disentangle_loss(msg)
    
    total_loss = loss_binary + loss_multi + alpha * loss_smooth + beta * loss_dis + gamma * loss_stat
    
    return total_loss, {
        'binary': loss_binary.item(),
        'multi': loss_multi.item(),
        'smooth': loss_smooth.item(),
        'dis': loss_dis.item(),
        'stat': loss_stat.item(),
        'total': total_loss.item()
    }


def train_epoch(model, data, train_mask, batch_size=256, optimizer=None,
                alpha=0.1, beta=0.1, gamma=0.05):
    """Train one epoch."""
    model.train()
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    
    indices = train_mask
    n = len(indices)
    total_losses = defaultdict(float)
    n_batches = 0
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        
        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        msg = data.msg[batch_idx]
        t = data.t[batch_idx]
        dt = data.dt[batch_idx]
        src_layer = data.src_layer[batch_idx]
        dst_layer = data.dst_layer[batch_idx]
        labels = data.label[batch_idx]
        attacks = data.attack[batch_idx]
        
        optimizer.zero_grad()
        
        if isinstance(model, DIDS_MFL):
            binary_logits, multi_logits, x_prev, x_curr = model(
                src, dst, msg, t, dt, src_layer, dst_layer
            )
            loss, loss_dict = compute_loss(
                binary_logits, multi_logits, labels, attacks,
                x_prev, x_curr, model.stat_disentangle, msg,
                alpha, beta, gamma
            )
        elif isinstance(model, TGNBaseline):
            binary_logits, multi_logits, x_prev, x_curr = model(
                src, dst, msg, t, dt, src_layer, dst_layer
            )
            loss_binary = F.cross_entropy(binary_logits, labels)
            loss_multi = F.cross_entropy(multi_logits, attacks)
            loss = loss_binary + loss_multi
            loss_dict = {'binary': loss_binary.item(), 'multi': loss_multi.item(), 
                        'total': loss.item()}
        else:
            binary_logits, multi_logits = model(msg)
            loss_binary = F.cross_entropy(binary_logits, labels)
            loss_multi = F.cross_entropy(multi_logits, attacks)
            loss = loss_binary + loss_multi
            loss_dict = {'binary': loss_binary.item(), 'multi': loss_multi.item(), 
                        'total': loss.item()}
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        for k, v in loss_dict.items():
            total_losses[k] += v
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model, data, eval_mask, batch_size=512):
    """Evaluate model on given data mask."""
    model.eval()
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    
    all_binary_preds = []
    all_binary_probs = []
    all_multi_preds = []
    all_multi_probs = []
    all_labels = []
    all_attacks = []
    all_reps = []
    
    indices = eval_mask
    n = len(indices)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        
        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        msg = data.msg[batch_idx]
        t = data.t[batch_idx]
        dt = data.dt[batch_idx]
        src_layer = data.src_layer[batch_idx]
        dst_layer = data.dst_layer[batch_idx]
        labels = data.label[batch_idx]
        attacks = data.attack[batch_idx]
        
        if isinstance(model, DIDS_MFL):
            binary_logits, multi_logits, reps, _, _ = model(
                src, dst, msg, t, dt, src_layer, dst_layer, return_reps=True
            )
            all_reps.append(reps.cpu())
        elif isinstance(model, TGNBaseline):
            binary_logits, multi_logits, _, _ = model(
                src, dst, msg, t, dt, src_layer, dst_layer
            )
        else:
            binary_logits, multi_logits = model(msg)
        
        all_binary_preds.append(binary_logits.argmax(dim=-1).cpu())
        all_binary_probs.append(F.softmax(binary_logits, dim=-1)[:, 1].cpu())
        all_multi_preds.append(multi_logits.argmax(dim=-1).cpu())
        all_multi_probs.append(F.softmax(multi_logits, dim=-1).cpu())
        all_labels.append(labels.cpu())
        all_attacks.append(attacks.cpu())
    
    binary_preds = torch.cat(all_binary_preds).numpy()
    binary_probs = torch.cat(all_binary_probs).numpy()
    multi_preds = torch.cat(all_multi_preds).numpy()
    multi_probs = torch.cat(all_multi_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    attacks = torch.cat(all_attacks).numpy()
    
    results = {}
    
    # Binary metrics
    results['binary_f1'] = f1_score(labels, binary_preds, average='binary')
    results['binary_precision'] = precision_score(labels, binary_preds, average='binary')
    results['binary_recall'] = recall_score(labels, binary_preds, average='binary')
    results['binary_accuracy'] = accuracy_score(labels, binary_preds)
    try:
        results['binary_auc'] = roc_auc_score(labels, binary_probs)
    except:
        results['binary_auc'] = 0.0
    
    # Multi-class metrics
    results['multi_f1_macro'] = f1_score(attacks, multi_preds, average='macro')
    results['multi_f1_weighted'] = f1_score(attacks, multi_preds, average='weighted')
    results['multi_accuracy'] = accuracy_score(attacks, multi_preds)
    
    # Per-attack F1
    attack_names = get_attack_names()
    per_attack_f1 = f1_score(attacks, multi_preds, average=None)
    results['per_attack_f1'] = {attack_names.get(i, f'class_{i}'): float(f) 
                                 for i, f in enumerate(per_attack_f1)}
    
    if len(all_reps) > 0:
        results['representations'] = torch.cat(all_reps).numpy()
    
    results['labels'] = labels
    results['attacks'] = attacks
    results['binary_preds'] = binary_preds
    results['multi_preds'] = multi_preds
    
    return results


def get_attack_names():
    """Map attack indices to names for NF-UNSW-NB15."""
    return {
        0: 'Analysis',
        1: 'Backdoor',
        2: 'Benign',
        3: 'DoS',
        4: 'Exploits',
        5: 'Fuzzers',
        6: 'Generic',
        7: 'Reconnaissance',
        8: 'Shellcode',
        9: 'Worms'
    }


def get_few_shot_attack_types(data):
    """Identify few-shot attack types (< 500 samples)."""
    attack_counts = Counter(data.attack.numpy().tolist())
    few_shot = {k: v for k, v in attack_counts.items() if v < 500 and k != 2}
    return few_shot


def prepare_data_splits(data, train_ratio=0.7, val_ratio=0.15):
    """Split data chronologically (time-based split)."""
    n = len(data.src)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_mask = np.arange(0, train_end)
    val_mask = np.arange(train_end, val_end)
    test_mask = np.arange(val_end, n)
    
    return train_mask, val_mask, test_mask


def prepare_unknown_attack_split(data, unknown_attack_id, train_ratio=0.7, val_ratio=0.15):
    """
    For unknown attack evaluation: remove unknown_attack_id from training,
    but include it in test.
    """
    n = len(data.src)
    attacks = data.attack.numpy()
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Remove unknown attack from train
    train_indices = np.arange(0, train_end)
    train_mask = train_indices[attacks[train_indices] != unknown_attack_id]
    
    val_indices = np.arange(train_end, val_end)
    val_mask = val_indices[attacks[val_indices] != unknown_attack_id]
    
    # Test includes all (including unknown)
    test_mask = np.arange(val_end, n)
    
    return train_mask, val_mask, test_mask


def prepare_few_shot_split(data, few_shot_attack_id, n_shots=5, train_ratio=0.7, val_ratio=0.15):
    """
    For few-shot evaluation: keep only n_shots of few_shot_attack_id in training.
    """
    n = len(data.src)
    attacks = data.attack.numpy()
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_indices = np.arange(0, train_end)
    
    # Find indices of few-shot attack in train
    few_shot_in_train = train_indices[attacks[train_indices] == few_shot_attack_id]
    other_in_train = train_indices[attacks[train_indices] != few_shot_attack_id]
    
    # Keep only n_shots
    if len(few_shot_in_train) > n_shots:
        few_shot_keep = np.random.choice(few_shot_in_train, n_shots, replace=False)
    else:
        few_shot_keep = few_shot_in_train
    
    train_mask = np.sort(np.concatenate([other_in_train, few_shot_keep]))
    val_mask = np.arange(train_end, val_end)
    test_mask = np.arange(val_end, n)
    
    return train_mask, val_mask, test_mask


if __name__ == '__main__':
    print("DIDS-MFL module loaded successfully.")
    print("Components: StatisticalDisentanglement, MemoryModule, RepresentationalDisentanglement,")
    print("            GraphDiffusion, MultiScaleFusion, DIDS_MFL")
