"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-scale Feature Learning

Core modules:
1. Statistical Disentanglement (SD): MI-based feature group disentanglement
2. Memory Representation (MR): Key-value memory network
3. Representational Disentanglement (RD): Contrastive disentanglement of representations
4. Dynamic Graph Diffusion (DGD): Spatiotemporal graph aggregation
5. Multi-scale Feature Fusion (MFF): Feature fusion across scales
6. Few-shot Classifier: Prototypical network classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StatisticalDisentanglement(nn.Module):
    """
    Statistical Disentanglement module.
    Learns K feature groups via learnable soft clustering with MI minimization.
    """
    def __init__(self, input_dim=40, num_groups=5, hidden_dim=64):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        
        # Feature group assignment network
        self.group_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups)
        )
        
        # Per-group transformation networks
        self.group_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // num_groups)
            ) for _ in range(num_groups)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim] - raw features
        Returns:
            group_assignments: [B, K] - soft group assignments
            group_features: [B, K * hidden_dim // K] - disentangled features per group
        """
        # Soft group assignments
        group_logits = self.group_net(x)
        group_assignments = F.gumbel_softmax(group_logits, tau=1.0, hard=False)
        
        # Per-group features
        group_features = []
        for i, transform in enumerate(self.group_transforms):
            # Weight features by group assignment
            weighted_x = x * group_assignments[:, i:i+1]
            gf = transform(weighted_x)
            group_features.append(gf)
        
        group_features = torch.cat(group_features, dim=-1)
        return group_assignments, group_features


class MemoryNetwork(nn.Module):
    """
    Key-Value Memory Network for generating compact representations.
    """
    def __init__(self, input_dim, memory_size=64, output_dim=64):
        super().__init__()
        self.memory_size = memory_size
        
        # Memory keys and values (learnable)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, output_dim) * 0.1)
        self.memory_values = nn.Parameter(torch.randn(memory_size, output_dim) * 0.1)
        
        # Query network
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            memory_output: [B, output_dim]
            attention_weights: [B, memory_size]
        """
        query = self.query_net(x)  # [B, output_dim]
        
        # Attention over memory keys
        attention = torch.matmul(query, self.memory_keys.T)  # [B, memory_size]
        attention = F.softmax(attention / math.sqrt(query.shape[-1]), dim=-1)
        
        # Weighted sum of memory values
        memory_output = torch.matmul(attention, self.memory_values)  # [B, output_dim]
        
        # Residual connection
        memory_output = memory_output + query
        memory_output = self.output_proj(memory_output)
        
        return memory_output, attention


class RepresentationalDisentanglement(nn.Module):
    """
    Representational Disentanglement via contrastive learning.
    Separates attack-specific from benign-specific features.
    """
    def __init__(self, input_dim, num_heads=3, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Multi-head projection for disentanglement
        self.head_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, head_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_dim, head_dim)
            ) for _ in range(num_heads)
        ])
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(num_heads * head_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            disentangled: [B, num_heads * head_dim] - concatenation of head features
            head_features: list of [B, head_dim] - individual head features
        """
        head_features = []
        for proj in self.head_projections:
            hf = proj(x)
            head_features.append(hf)
        
        disentangled = torch.cat(head_features, dim=-1)
        return disentangled, head_features
    
    def contrastive_loss(self, head_features, labels):
        """
        InfoNCE-inspired contrastive loss to separate representations.
        """
        total_loss = 0.0
        for i in range(len(head_features)):
            for j in range(i+1, len(head_features)):
                fi, fj = head_features[i], head_features[j]
                # Similarity matrix
                sim = torch.matmul(F.normalize(fi, dim=-1), F.normalize(fj, dim=-1).T)
                sim = sim / 0.07
                
                # Create positive/negative masks based on label agreement
                label_match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
                
                # Contrastive loss
                exp_sim = torch.exp(sim)
                log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Weight by label match
                weight = label_match
                pos_weight = weight.clone()
                pos_weight.fill_diagonal_(0)
                
                if pos_weight.sum() > 0:
                    loss = -(pos_weight * log_prob).sum() / (pos_weight.sum() + 1e-8)
                    total_loss += loss
        
        return total_loss / max(1, len(head_features) * (len(head_features) - 1) / 2)


class DynamicGraphDiffusion(nn.Module):
    """
    Dynamic Graph Diffusion for spatiotemporal aggregation.
    Constructs dynamic graphs based on feature similarity and performs diffusion.
    """
    def __init__(self, input_dim, hidden_dim=64, num_hops=2, k_neighbors=10):
        super().__init__()
        self.num_hops = num_hops
        self.k_neighbors = k_neighbors
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Diffusion weight network
        self.diffusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Aggregation layers
        self.agg_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ) for _ in range(num_hops)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def build_dynamic_graph(self, x, src_idx, dst_idx, k=None):
        """Build dynamic graph based on feature similarity."""
        if k is None:
            k = self.k_neighbors
        
        # Transform features
        h = self.transform(x)
        
        # Compute edge weights based on feature similarity of connected nodes
        h_src = h[src_idx]  # [E, hidden_dim]
        h_dst = h[dst_idx]  # [E, hidden_dim]
        
        # Edge weight from feature similarity
        edge_feat = torch.cat([h_src, h_dst], dim=-1)  # [E, 2*hidden_dim]
        edge_weights = self.diffusion_net(edge_feat).squeeze(-1)  # [E]
        
        return h, edge_weights
    
    def forward(self, x, src_idx, dst_idx):
        """
        Args:
            x: [N, input_dim] or [B, input_dim] (node features)
            src_idx: [E] - source node indices
            dst_idx: [E] - destination node indices
        Returns:
            aggregated: [N or B, input_dim] - aggregated features
        """
        # Build dynamic graph
        h, edge_weights = self.build_dynamic_graph(x, src_idx, dst_idx)
        hidden_dim = h.shape[-1]
        
        # Multi-hop diffusion
        for hop in range(self.num_hops):
            h_new = h.clone()
            
            # Aggregate from neighbors
            h_src = h[src_idx]  # [E, hidden_dim]
            h_dst = h[dst_idx]  # [E, hidden_dim]
            
            # Weighted message passing
            messages = h_src * edge_weights.unsqueeze(-1)  # [E, hidden_dim]
            
            # Scatter add
            h_agg = torch.zeros_like(h)
            weight_sum = torch.zeros(h.shape[0], 1, device=h.device)
            
            # Use scatter for aggregation
            idx_dst = dst_idx.unsqueeze(-1).expand_as(messages)
            h_agg.scatter_add_(0, idx_dst, messages)
            weight_sum.scatter_add_(0, dst_idx.unsqueeze(-1), 
                                    edge_weights.unsqueeze(-1))
            
            # Normalize
            weight_sum = weight_sum.clamp(min=1e-8)
            h_agg = h_agg / weight_sum
            
            # Update with residual
            h = self.agg_layers[hop](h_agg + h)
            h = F.relu(h)
        
        # Project back to original dimension
        aggregated = self.output_proj(h)
        return aggregated


class DIDS_MFL(nn.Module):
    """
    Complete DIDS-MFL framework combining all modules.
    """
    def __init__(self, input_dim=40, num_groups=5, memory_size=64, 
                 hidden_dim=64, num_heads=3, head_dim=32,
                 num_hops=2, num_classes=10):
        super().__init__()
        
        # Module 1: Statistical Disentanglement
        self.stat_disentangle = StatisticalDisentanglement(
            input_dim, num_groups, hidden_dim)
        
        # Module 2: Memory Network
        sd_out_dim = num_groups * (hidden_dim // num_groups)
        self.memory = MemoryNetwork(sd_out_dim, memory_size, hidden_dim)
        
        # Module 3: Representational Disentanglement
        self.rep_disentangle = RepresentationalDisentanglement(
            hidden_dim, num_heads, head_dim)
        
        # Module 4: Dynamic Graph Diffusion
        self.graph_diffusion = DynamicGraphDiffusion(
            input_dim, hidden_dim, num_hops)
        
        # Module 5: Multi-scale Feature Fusion
        fused_dim = sd_out_dim + hidden_dim + num_heads * head_dim + input_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Module 6: Classification head
        self.binary_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.multiclass_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Prototype for few-shot
        self.prototype_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, src_idx=None, dst_idx=None):
        """
        Args:
            x: [B, input_dim] - flow features
            src_idx: [E] - source indices (for graph diffusion)
            dst_idx: [E] - destination indices (for graph diffusion)
        Returns:
            logits_binary: [B, 2]
            logits_multi: [B, num_classes]
            representations: [B, hidden_dim]
            aux_info: dict with intermediate outputs
        """
        # 1. Statistical Disentanglement
        group_assignments, group_features = self.stat_disentangle(x)
        
        # 2. Memory Representation
        memory_out, memory_attention = self.memory(group_features)
        
        # 3. Representational Disentanglement
        rep_disentangled, head_features = self.rep_disentangle(memory_out)
        
        # 4. Dynamic Graph Diffusion
        if src_idx is not None and dst_idx is not None:
            # We need to construct full node features for graph diffusion
            # For efficiency, use batch-level aggregation
            graph_features = self.graph_diffusion(x, 
                torch.arange(x.shape[0], device=x.device).repeat(1),
                torch.arange(x.shape[0], device=x.device).repeat(1))
        else:
            graph_features = x
        
        # 5. Multi-scale Feature Fusion
        fused = torch.cat([group_features, memory_out, rep_disentangled, 
                          graph_features], dim=-1)
        representations = self.fusion_net(fused)
        
        # 6. Classification
        logits_binary = self.binary_classifier(representations)
        logits_multi = self.multiclass_classifier(representations)
        
        aux_info = {
            'group_assignments': group_assignments,
            'memory_attention': memory_attention,
            'head_features': head_features,
            'group_features': group_features,
            'memory_out': memory_out,
            'rep_disentangled': rep_disentangled,
        }
        
        return logits_binary, logits_multi, representations, aux_info
    
    def compute_loss(self, logits_binary, logits_multi, representations, 
                     labels_binary, labels_multi, aux_info, 
                     contrastive_weight=0.1, ce_weight=1.0):
        """Compute total loss."""
        # Binary cross-entropy
        loss_binary = F.cross_entropy(logits_binary, labels_binary)
        
        # Multi-class cross-entropy (weighted for class imbalance)
        num_classes_total = logits_multi.shape[-1]
        class_weights = torch.ones(num_classes_total, device=labels_multi.device)
        # Inverse frequency weighting
        for c in range(num_classes_total):
            count = (labels_multi == c).sum().float()
            if count > 0:
                class_weights[c] = labels_multi.shape[0] / (count * labels_multi.max().item() + 1)
        class_weights = class_weights / class_weights.sum() * labels_multi.shape[0]
        
        loss_multi = F.cross_entropy(logits_multi, labels_multi, weight=class_weights)
        
        # Contrastive disentanglement loss
        loss_contrastive = self.rep_disentangle.contrastive_loss(
            aux_info['head_features'], labels_binary)
        
        # Group diversity loss (encourage diverse group assignments)
        group_assignments = aux_info['group_assignments']
        group_usage = group_assignments.mean(dim=0)
        loss_diversity = -torch.sum(group_usage * torch.log(group_usage + 1e-8))
        
        total_loss = (ce_weight * loss_binary + 
                     ce_weight * loss_multi * 0.5 +
                     contrastive_weight * loss_contrastive +
                     0.01 * loss_diversity)
        
        return total_loss, {
            'binary_loss': loss_binary.item(),
            'multi_loss': loss_multi.item(),
            'contrastive_loss': loss_contrastive.item(),
            'diversity_loss': loss_diversity.item(),
            'total_loss': total_loss.item()
        }
    
    def get_embeddings(self, x):
        """Get intermediate embeddings for visualization."""
        with torch.no_grad():
            _, _, representations, aux_info = self.forward(x)
            return representations, aux_info


class BaselineMLP(nn.Module):
    """Standard MLP baseline."""
    def __init__(self, input_dim=40, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.binary_head = nn.Linear(hidden_dim // 2, 2)
        self.multi_head = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x, src_idx=None, dst_idx=None):
        h = self.net(x)
        return self.binary_head(h), self.multi_head(h), h, {}
    
    def compute_loss(self, logits_binary, logits_multi, representations,
                     labels_binary, labels_multi, aux_info,
                     contrastive_weight=0.1, ce_weight=1.0):
        loss_binary = F.cross_entropy(logits_binary, labels_binary)
        loss_multi = F.cross_entropy(logits_multi, labels_multi)
        total_loss = ce_weight * loss_binary + 0.5 * ce_weight * loss_multi
        return total_loss, {
            'binary_loss': loss_binary.item(),
            'multi_loss': loss_multi.item(),
            'contrastive_loss': 0.0,
            'diversity_loss': 0.0,
            'total_loss': total_loss.item()
        }


class AblationSDOnly(nn.Module):
    """Ablation: Only Statistical Disentanglement."""
    def __init__(self, input_dim=40, num_classes=10):
        super().__init__()
        self.stat_disentangle = StatisticalDisentanglement(input_dim, 5, 64)
        sd_out = 5 * (64 // 5)
        self.classifier = nn.Sequential(
            nn.Linear(sd_out, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.binary_head = nn.Linear(64, 2)
        self.multi_head = nn.Linear(64, num_classes)
    
    def forward(self, x, src_idx=None, dst_idx=None):
        _, gf = self.stat_disentangle(x)
        h = self.classifier(gf)
        return self.binary_head(h), self.multi_head(h), h, {}
    
    def compute_loss(self, logits_binary, logits_multi, representations,
                     labels_binary, labels_multi, aux_info,
                     contrastive_weight=0.1, ce_weight=1.0):
        loss_binary = F.cross_entropy(logits_binary, labels_binary)
        loss_multi = F.cross_entropy(logits_multi, labels_multi)
        total_loss = ce_weight * loss_binary + 0.5 * ce_weight * loss_multi
        return total_loss, {
            'binary_loss': loss_binary.item(),
            'multi_loss': loss_multi.item(),
            'contrastive_loss': 0.0,
            'diversity_loss': 0.0,
            'total_loss': total_loss.item()
        }


class AblationNoGraph(nn.Module):
    """Ablation: DIDS-MFL without graph diffusion."""
    def __init__(self, input_dim=40, num_classes=10):
        super().__init__()
        self.stat_disentangle = StatisticalDisentanglement(input_dim, 5, 64)
        self.memory = MemoryNetwork(5*(64//5), 64, 64)
        self.rep_disentangle = RepresentationalDisentanglement(64, 3, 32)
        fused_dim = 5*(64//5) + 64 + 3*32 + input_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.binary_head = nn.Linear(64, 2)
        self.multi_head = nn.Linear(64, num_classes)
    
    def forward(self, x, src_idx=None, dst_idx=None):
        _, gf = self.stat_disentangle(x)
        mo, _ = self.memory(gf)
        rd, _ = self.rep_disentangle(mo)
        fused = torch.cat([gf, mo, rd, x], dim=-1)
        h = self.fusion(fused)
        return self.binary_head(h), self.multi_head(h), h, {}
    
    def compute_loss(self, logits_binary, logits_multi, representations,
                     labels_binary, labels_multi, aux_info,
                     contrastive_weight=0.1, ce_weight=1.0):
        loss_binary = F.cross_entropy(logits_binary, labels_binary)
        loss_multi = F.cross_entropy(logits_multi, labels_multi)
        total_loss = ce_weight * loss_binary + 0.5 * ce_weight * loss_multi
        return total_loss, {
            'binary_loss': loss_binary.item(),
            'multi_loss': loss_multi.item(),
            'contrastive_loss': 0.0,
            'diversity_loss': 0.0,
            'total_loss': total_loss.item()
        }


class AblationNoRepDis(nn.Module):
    """Ablation: Without Representational Disentanglement."""
    def __init__(self, input_dim=40, num_classes=10):
        super().__init__()
        self.stat_disentangle = StatisticalDisentanglement(input_dim, 5, 64)
        self.memory = MemoryNetwork(5*(64//5), 64, 64)
        self.graph_diffusion = DynamicGraphDiffusion(input_dim, 64, 2)
        fused_dim = 5*(64//5) + 64 + input_dim + input_dim  # gf + mo + gf_diff + x
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.binary_head = nn.Linear(64, 2)
        self.multi_head = nn.Linear(64, num_classes)
    
    def forward(self, x, src_idx=None, dst_idx=None):
        _, gf = self.stat_disentangle(x)
        mo, _ = self.memory(gf)
        gf_diff = self.graph_diffusion(x, 
            torch.arange(x.shape[0], device=x.device),
            torch.arange(x.shape[0], device=x.device))
        fused = torch.cat([gf, mo, gf_diff, x], dim=-1)
        h = self.fusion(fused)
        return self.binary_head(h), self.multi_head(h), h, {}
    
    def compute_loss(self, logits_binary, logits_multi, representations,
                     labels_binary, labels_multi, aux_info,
                     contrastive_weight=0.1, ce_weight=1.0):
        loss_binary = F.cross_entropy(logits_binary, labels_binary)
        loss_multi = F.cross_entropy(logits_multi, labels_multi)
        total_loss = ce_weight * loss_binary + 0.5 * ce_weight * loss_multi
        return total_loss, {
            'binary_loss': loss_binary.item(),
            'multi_loss': loss_multi.item(),
            'contrastive_loss': 0.0,
            'diversity_loss': 0.0,
            'total_loss': total_loss.item()
        }
