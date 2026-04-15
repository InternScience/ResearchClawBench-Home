"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-scale Feature Learning

This module implements the core model architecture including:
1. Statistical Disentanglement Module (SDM) - MI-based feature separation
2. Representational Disentanglement Module (RDM) - Regularized latent space
3. Dynamic Graph Diffusion (DGD) - Non-linear temporal graph diffusion
4. Multi-scale Feature Fusion (MFF) - Hierarchical representation fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class StatisticalDisentanglement(nn.Module):
    """
    Statistical Disentanglement Module (SDM)
    Uses non-parametric mutual information optimization to separate entangled
    statistical flow features into differentiated components.
    """
    def __init__(self, input_dim, hidden_dim, num_factors=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        
        # Factor assignment network
        self.factor_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors),
        )
        
        # Factor-specific feature extractors
        self.factor_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // num_factors),
            ) for _ in range(num_factors)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] raw features
        Returns:
            disentangled: [batch, num_factors * factor_dim]
            factor_weights: [batch, num_factors] soft assignment
        """
        # Compute factor importance weights
        factor_logits = self.factor_encoder(x)
        factor_weights = F.softmax(factor_logits, dim=-1)  # [B, K]
        
        # Extract factor-specific features
        factor_features = []
        for k in range(self.num_factors):
            fk = self.factor_extractors[k](x)  # [B, d_k]
            factor_features.append(fk)
        
        # Weighted combination
        stacked = torch.stack(factor_features, dim=1)  # [B, K, d_k]
        # Expand weights for broadcasting
        w = factor_weights.unsqueeze(-1)  # [B, K, 1]
        weighted = stacked * w  # [B, K, d_k]
        disentangled = weighted.view(x.size(0), -1)  # [B, K*d_k]
        
        return disentangled, factor_weights
    
    def mi_regularization(self, factor_weights):
        """
        Mutual information based regularization to encourage independence
        between factors. Minimizes total correlation approximation.
        """
        # Compute pairwise correlation of factor weights
        # E[w_i * w_j] - E[w_i]*E[w_j]
        batch_size = factor_weights.size(0)
        mean_w = factor_weights.mean(dim=0, keepdim=True)  # [1, K]
        centered = factor_weights - mean_w  # [B, K]
        corr = torch.matmul(centered.t(), centered) / batch_size  # [K, K]
        # Off-diagonal penalty
        mask = ~torch.eye(corr.size(0), device=corr.device).bool()
        tc_loss = (corr[mask] ** 2).mean()
        return tc_loss


class RepresentationalDisentanglement(nn.Module):
    """
    Representational Disentanglement Module (RDM)
    Applies regularization on learned representations to highlight
    attack-specific features with smaller cross-feature correlations.
    """
    def __init__(self, input_dim, output_dim, num_classes=10):
        super().__init__()
        self.representation_net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.num_classes = num_classes
        
    def forward(self, x):
        rep = self.representation_net(x)
        return rep
    
    def disentanglement_loss(self, rep, labels=None):
        """
        Regularization loss to reduce inter-feature correlations
        and encourage class-discriminative representations.
        """
        # Normalize representations
        rep_norm = F.normalize(rep, p=2, dim=-1)
        
        # Pearson correlation matrix of features
        batch_size = rep_norm.size(0)
        mean_r = rep_norm.mean(dim=0, keepdim=True)
        centered = rep_norm - mean_r
        corr = torch.matmul(centered.t(), centered) / (batch_size - 1)
        
        # Penalize off-diagonal correlations
        dim = corr.size(0)
        mask = ~torch.eye(dim, device=corr.device).bool()
        decorrelation_loss = (corr[mask] ** 2).mean()
        
        return decorrelation_loss


class DynamicGraphDiffusion(MessagePassing):
    """
    Dynamic Graph Diffusion layer for spatiotemporal aggregation.
    Implements non-linear diffusion on evolving graph structures.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Attention mechanism for dynamic edge weighting
        self.att_query = nn.Linear(in_channels, heads * out_channels)
        self.att_key = nn.Linear(in_channels, heads * out_channels)
        self.att_value = nn.Linear(in_channels, heads * out_channels)
        
        # Temporal encoding
        self.temporal_proj = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, heads),
        )
        
        # Output projection
        self.out_proj = nn.Linear(heads * out_channels, out_channels)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_query.weight)
        nn.init.xavier_uniform_(self.att_key.weight)
        nn.init.xavier_uniform_(self.att_value.weight)
        
    def forward(self, x, edge_index, t=None):
        """
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            t: [E] edge timestamps (optional)
        """
        N = x.size(0)
        H = self.heads
        D = self.out_channels
        
        # Project to attention space
        Q = self.att_query(x).view(N, H, D)  # [N, H, D]
        K = self.att_key(x).view(N, H, D)    # [N, H, D]
        V = self.att_value(x).view(N, H, D)  # [N, H, D]
        
        # Get source and target indices
        row, col = edge_index[0], edge_index[1]  # [E]
        
        # Compute attention scores
        q_src = Q[row]  # [E, H, D]
        k_dst = K[col]  # [E, H, D]
        alpha = (q_src * k_dst).sum(dim=-1) / (D ** 0.5)  # [E, H]
        
        # Add temporal component if available
        if t is not None:
            t_emb = self.temporal_proj(t.float().unsqueeze(-1))  # [E, H]
            alpha = alpha + t_emb
        
        # Softmax normalization per target node
        alpha = softmax(alpha, col, num_nodes=N)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate values
        v_dst = V[col]  # [E, H, D]
        out = alpha.unsqueeze(-1) * v_dst  # [E, H, D]
        
        # Sum aggregation per target
        result = torch.zeros(N, H, D, device=x.device)
        result.index_add_(0, col, out)
        
        # Reshape and project
        result = result.view(N, H * D)
        result = self.out_proj(result)
        
        return result


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale Feature Fusion module for combining representations
    at different granularities, enhancing few-shot learning capability.
    """
    def __init__(self, base_dim, scales=3):
        super().__init__()
        self.scales = scales
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.LayerNorm(base_dim),
                nn.ReLU(),
            ) for _ in range(scales)
        ])
        self.fusion_gate = nn.Sequential(
            nn.Linear(base_dim * scales, scales),
            nn.Softmax(dim=-1),
        )
        self.output_proj = nn.Linear(base_dim, base_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, base_dim] input features
        Returns:
            fused: [batch, base_dim] multi-scale fused features
        """
        scale_features = []
        for i in range(self.scales):
            sf = self.scale_convs[i](x)
            scale_features.append(sf)
        
        # Concatenate for gate computation
        concat = torch.cat(scale_features, dim=-1)  # [B, base_dim*scales]
        gates = self.fusion_gate(concat)  # [B, scales]
        
        # Weighted sum
        stacked = torch.stack(scale_features, dim=1)  # [B, scales, base_dim]
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)  # [B, base_dim]
        fused = self.output_proj(fused)
        
        return fused


class DIDS_MFL(nn.Module):
    """
    Complete DIDS-MFL model integrating all components:
    - Statistical Disentanglement
    - Representational Disentanglement  
    - Dynamic Graph Diffusion
    - Multi-scale Feature Fusion
    """
    def __init__(self, feature_dim=40, hidden_dim=128, num_classes=10, 
                 num_factors=8, num_gnn_layers=3, heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Statistical Disentanglement
        self.sdm = StatisticalDisentanglement(hidden_dim, hidden_dim, num_factors)
        
        # Representational Disentanglement
        sdm_out_dim = hidden_dim // num_factors * num_factors
        self.rdm = RepresentationalDisentanglement(sdm_out_dim, hidden_dim, num_classes)
        
        # Dynamic Graph Diffusion layers
        self.gnn_layers = nn.ModuleList([
            DynamicGraphDiffusion(hidden_dim, hidden_dim, heads=heads)
            for _ in range(num_gnn_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Multi-scale Feature Fusion
        self.mff = MultiScaleFeatureFusion(hidden_dim, scales=3)
        
        # Classification heads
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Multi-class classification head
        self.multiclass_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
    def forward(self, x, edge_index, t=None):
        """
        Args:
            x: [N, feature_dim] node/edge features
            edge_index: [2, E] graph edges
            t: [E] timestamps
        Returns:
            binary_logit: [N, 1]
            multiclass_logit: [N, num_classes]
            factor_weights: [N, num_factors]
            rep: [N, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Statistical disentanglement
        h_disentangled, factor_weights = self.sdm(h)
        
        # Representational disentanglement
        rep = self.rdm(h_disentangled)
        
        # Graph diffusion
        h_graph = rep
        for gnn, norm in zip(self.gnn_layers, self.norm_layers):
            h_graph = gnn(h_graph, edge_index, t)
            h_graph = norm(h_graph + h_graph)  # residual + norm
            h_graph = F.relu(h_graph)
        
        # Multi-scale fusion
        h_fused = self.mff(h_graph)
        
        # Classification
        binary_logit = self.binary_head(h_fused)
        multiclass_logit = self.multiclass_head(h_fused)
        
        return binary_logit, multiclass_logit, factor_weights, rep
    
    def compute_loss(self, binary_logit, multiclass_logit, factor_weights, rep,
                     binary_labels, multiclass_labels, alpha_mi=0.1, alpha_dec=0.05):
        """
        Compute combined loss with disentanglement regularization.
        """
        # Binary classification loss
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit.squeeze(-1), binary_labels.float()
        )
        
        # Multi-class classification loss
        multiclass_loss = F.cross_entropy(multiclass_logit, multiclass_labels)
        
        # MI regularization for statistical disentanglement
        mi_loss = self.sdm.mi_regularization(factor_weights)
        
        # Decorrelation loss for representational disentanglement
        dec_loss = self.rdm.disentanglement_loss(rep)
        
        total_loss = (binary_loss + multiclass_loss + 
                     alpha_mi * mi_loss + alpha_dec * dec_loss)
        
        return total_loss, {
            'binary': binary_loss.item(),
            'multiclass': multiclass_loss.item(),
            'mi_reg': mi_loss.item(),
            'dec_reg': dec_loss.item(),
        }
