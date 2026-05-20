"""
Unified Autoregressive Framework with Decoupled Visual Encoding
================================================================
This module implements a unified Transformer architecture that decouples visual encoding
to perform both multimodal understanding (VQA) and visual generation (text-to-image)
within a single framework.

Key innovations:
1. Decoupled Visual Encoders: Separate encoders for understanding and generation
2. Cross-Encoder Alignment: Bridge the two visual representations
3. Adaptive Task Routing: Dynamic routing based on input task type
4. Unified Autoregressive Backbone: Shared Transformer for both modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class FrameworkConfig:
    """Configuration for the unified framework."""
    # Model dimensions
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    feedforward_dim: int = 3072
    dropout: float = 0.1
    
    # Visual encoder configs
    understanding_encoder_dim: int = 512
    generation_encoder_dim: int = 512
    codebook_size: int = 8192
    patch_size: int = 16
    image_size: int = 224
    
    # Text tokenizer
    vocab_size: int = 32000
    max_seq_len: int = 512
    
    # Alignment
    alignment_dim: int = 768
    temperature: float = 0.07
    
    # Task types
    num_task_types: int = 3  # understanding, generation, mixed


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for Transformer."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # Compute rotation frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
        # Precompute rotation matrices
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        cos = self.cos_cached[:seq_len].unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0)
        
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with QK-Norm for training stability."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # QK-Norm for training stability (from Chameleon)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation."""
    
    def __init__(self, hidden_dim: int, feedforward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, feedforward_dim, bias=False)
        self.w2 = nn.Linear(feedforward_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, feedforward_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x @ W1) * silu(x @ W3)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer Block with Swin-style normalization for stability."""
    
    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = FeedForward(hidden_dim, feedforward_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Swin-style normalization (from Chameleon-34B)
        h = x + self.dropout(self.attention(self.attention_norm(x), mask))
        output = h + self.dropout(self.feedforward(self.ff_norm(h)))
        return output


class VisualUnderstandingEncoder(nn.Module):
    """
    Visual Understanding Encoder (VUE)
    Optimized for multimodal understanding tasks like VQA and captioning.
    Uses a CLIP-style architecture with patch-based encoding.
    """
    
    def __init__(self, config: FrameworkConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, config.understanding_encoder_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Positional embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.understanding_encoder_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.understanding_encoder_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.understanding_encoder_dim,
                config.num_heads,
                config.feedforward_dim,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.understanding_encoder_dim)
        
        # Projection to unified space
        self.projector = nn.Sequential(
            nn.Linear(config.understanding_encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, C
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Project to unified space
        return self.projector(x)


class VisualGenerationEncoder(nn.Module):
    """
    Visual Generation Encoder (VGE)
    A VQGAN-style tokenizer for image tokenization and generation.
    Converts images to discrete tokens for autoregressive generation.
    """
    
    def __init__(self, config: FrameworkConfig):
        super().__init__()
        self.config = config
        
        # Encoder (ConvNet-based)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, config.generation_encoder_dim, kernel_size=3, stride=1, padding=1),
        )
        
        # Vector Quantization
        self.codebook = nn.Embedding(config.codebook_size, config.generation_encoder_dim)
        
        # Decoder (ConvNet-based)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.generation_encoder_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Projection to unified space
        self.projector = nn.Sequential(
            nn.Linear(config.generation_encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vector quantization with straight-through estimator."""
        # z: B, C, H, W
        z_permuted = z.permute(0, 2, 3, 1)  # B, H, W, C
        z_flat = z_permuted.reshape(-1, self.config.generation_encoder_dim)
        
        # Find nearest codebook vector
        distances = torch.cdist(z_flat.unsqueeze(0), self.codebook.weight.unsqueeze(0))
        indices = distances.argmin(dim=-1)
        
        # Get quantized vectors
        quantized = self.codebook(indices).reshape(z_permuted.shape)
        
        # Straight-through estimator
        quantized_st = z_permuted + (quantized - z_permuted).detach()
        
        # Compute loss
        commitment_loss = F.mse_loss(z_permuted, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z_permuted.detach())
        
        return quantized_st.permute(0, 3, 1, 2), indices.reshape(z.shape[0], z.shape[2], z.shape[3]), commitment_loss + codebook_loss
    
    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_quantized, indices, vq_loss = self.quantize(z)
        
        if return_tokens:
            # Return discrete tokens for autoregressive generation
            return indices, vq_loss
        
        # Decode for reconstruction
        x_recon = self.decoder(z_quantized)
        
        # Project to unified space
        z_projected = z_quantized.flatten(2).transpose(1, 2)
        z_projected = self.projector(z_projected)
        
        return x_recon, z_projected, vq_loss


class CrossEncoderAlignment(nn.Module):
    """
    Cross-Encoder Alignment module
    Bridges the two visual encoders (VUE and VGE) through cross-attention.
    """
    
    def __init__(self, config: FrameworkConfig):
        super().__init__()
        self.config = config
        
        # Cross-attention from understanding to generation
        self.cross_attn_u2g = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Cross-attention from generation to understanding
        self.cross_attn_g2u = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm_u = nn.LayerNorm(config.hidden_dim)
        self.norm_g = nn.LayerNorm(config.hidden_dim)
        
        # Feed-forward networks
        self.ff_u = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        self.ff_g = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
    
    def forward(
        self,
        understanding_features: torch.Tensor,
        generation_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: understanding attends to generation
        u_norm = self.norm_u(understanding_features)
        g_norm = self.norm_g(generation_features)
        
        u_cross, _ = self.cross_attn_u2g(u_norm, g_norm, g_norm)
        understanding_features = understanding_features + u_cross
        understanding_features = understanding_features + self.ff_u(self.norm_u(understanding_features))
        
        # Cross-attention: generation attends to understanding
        g_cross, _ = self.cross_attn_g2u(g_norm, u_norm, u_norm)
        generation_features = generation_features + g_cross
        generation_features = generation_features + self.ff_g(self.norm_g(generation_features))
        
        return understanding_features, generation_features


class UnifiedTransformerBackbone(nn.Module):
    """
    Unified Transformer Backbone
    Shared autoregressive Transformer that handles both understanding and generation.
    """
    
    def __init__(self, config: FrameworkConfig):
        super().__init__()
        self.config = config
        
        # Task type embedding
        self.task_embed = nn.Embedding(config.num_task_types, config.hidden_dim)
        
        # Combined input projection
        self.input_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.feedforward_dim,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Output heads
        self.understanding_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.generation_head = nn.Linear(config.hidden_dim, config.codebook_size)
    
    def forward(
        self,
        combined_features: torch.Tensor,
        task_type: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = combined_features.shape
        
        # Add task embedding
        task_emb = self.task_embed(torch.tensor([task_type], device=combined_features.device))
        x = combined_features + task_emb.unsqueeze(1)
        
        # Create causal mask if not provided
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            mask = ~mask
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Output heads
        understanding_logits = self.understanding_head(x)
        generation_logits = self.generation_head(x)
        
        return understanding_logits, generation_logits


class UnifiedFramework(nn.Module):
    """
    Unified Autoregressive Framework with Decoupled Visual Encoding
    ================================================================
    A single Transformer architecture that performs both multimodal understanding
    and visual generation through decoupled visual encoding.
    """
    
    def __init__(self, config: FrameworkConfig):
        super().__init__()
        self.config = config
        
        # Decoupled visual encoders
        self.understanding_encoder = VisualUnderstandingEncoder(config)
        self.generation_encoder = VisualGenerationEncoder(config)
        
        # Cross-encoder alignment
        self.cross_alignment = CrossEncoderAlignment(config)
        
        # Unified Transformer backbone
        self.backbone = UnifiedTransformerBackbone(config)
        
        # Text tokenizer (simple embedding for demonstration)
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
    
    def encode_image_for_understanding(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image using the understanding encoder."""
        return self.understanding_encoder(image)
    
    def encode_image_for_generation(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image using the generation encoder."""
        return self.generation_encoder(image)
    
    def align_encoders(
        self,
        understanding_features: torch.Tensor,
        generation_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align features from both encoders."""
        return self.cross_alignment(understanding_features, generation_features)
    
    def forward(
        self,
        image: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        task_type: str = "understanding",
        return_generation_tokens: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the unified framework.
        
        Args:
            image: Input image tensor (B, C, H, W)
            text_tokens: Optional text tokens for understanding tasks
            task_type: "understanding", "generation", or "mixed"
            return_generation_tokens: If True, return discrete tokens for generation
        
        Returns:
            Dictionary containing outputs based on task type
        """
        # Encode image with both encoders
        understanding_features = self.encode_image_for_understanding(image)
        
        if return_generation_tokens:
            # For generation: get discrete tokens
            generation_tokens, vq_loss = self.generation_encoder(image, return_tokens=True)
            # Get embeddings from codebook
            gen_embeds = self.generation_encoder.codebook(generation_tokens.flatten(1))  # B, seq_len, C
            # Pad or truncate to match understanding features length
            target_len = understanding_features.shape[1]
            gen_len = gen_embeds.shape[1]
            if gen_len > target_len:
                gen_embeds = gen_embeds[:, :target_len, :]
            elif gen_len < target_len:
                pad_size = target_len - gen_len
                gen_embeds = torch.cat([gen_embeds, torch.zeros(gen_embeds.shape[0], pad_size, gen_embeds.shape[2], device=gen_embeds.device)], dim=1)
            generation_features = self.generation_encoder.projector(gen_embeds)
        else:
            # For understanding: get continuous features
            _, generation_features, vq_loss = self.encode_image_for_generation(image)
            # Pad or truncate to match understanding features length
            target_len = understanding_features.shape[1]
            gen_len = generation_features.shape[1]
            if gen_len > target_len:
                generation_features = generation_features[:, :target_len, :]
            elif gen_len < target_len:
                pad_size = target_len - gen_len
                generation_features = torch.cat([generation_features, torch.zeros(generation_features.shape[0], pad_size, generation_features.shape[2], device=generation_features.device)], dim=1)
        
        # Align encoders
        understanding_features, generation_features = self.align_encoders(
            understanding_features, generation_features
        )
        
        # Combine features
        combined = torch.cat([understanding_features, generation_features], dim=-1)
        combined = self.backbone.input_proj(combined)
        
        # Task type mapping
        task_map = {"understanding": 0, "generation": 1, "mixed": 2}
        task_id = task_map.get(task_type, 0)
        
        # Forward through backbone
        understanding_logits, generation_logits = self.backbone(combined, task_id)
        
        result = {
            "understanding_logits": understanding_logits,
            "generation_logits": generation_logits,
            "vq_loss": vq_loss,
            "understanding_features": understanding_features,
            "generation_features": generation_features
        }
        
        if text_tokens is not None:
            # For understanding: compute text loss
            text_embeds = self.text_embed(text_tokens[:, :-1])
            combined_text = torch.cat([
                understanding_features.mean(dim=1, keepdim=True).expand(-1, text_embeds.size(1), -1),
                text_embeds
            ], dim=-1)
            text_logits, _ = self.backbone(combined_text, task_id)
            result["text_logits"] = text_logits
        
        return result
    
    def generate_image(
        self,
        text_tokens: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        guidance_scale: float = 7.5
    ) -> torch.Tensor:
        """
        Generate image tokens autoregressively.
        
        Args:
            text_tokens: Text tokens (B, T)
            max_length: Maximum sequence length for generation
            temperature: Sampling temperature
            guidance_scale: Classifier-free guidance scale
        
        Returns:
            Generated image tokens (B, max_length)
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Encode text
        text_embeds = self.text_embed(text_tokens)
        text_pooled = text_embeds.mean(dim=1)
        
        # Initialize generation tokens
        generated_tokens = []
        
        for i in range(max_length):
            # Create input sequence
            if len(generated_tokens) == 0:
                # Start with text features
                input_features = text_pooled.unsqueeze(1)
            else:
                # Concatenate previous tokens
                prev_tokens = torch.stack(generated_tokens, dim=1)
                prev_embeds = self.generation_encoder.codebook(prev_tokens)
                input_features = torch.cat([
                    text_pooled.unsqueeze(1).expand(-1, prev_embeds.size(1), -1),
                    prev_embeds
                ], dim=-1)
                input_features = self.backbone.input_proj(input_features)
            
            # Forward through backbone
            _, generation_logits = self.backbone(input_features, task_type=1)
            
            # Get next token logits
            next_token_logits = generation_logits[:, -1, :] / temperature
            
            # Apply guidance
            if guidance_scale > 1.0:
                # Unconditional forward
                _, uncond_logits = self.backbone(
                    torch.zeros_like(input_features), task_type=1
                )
                uncond_logits = uncond_logits[:, -1, :] / temperature
                next_token_logits = uncond_logits + guidance_scale * (next_token_logits - uncond_logits)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            generated_tokens.append(next_token)
        
        return torch.stack(generated_tokens, dim=1)
    
    def decode_image_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode image tokens to image pixels."""
        batch_size = tokens.shape[0]
        device = tokens.device
        
        # Get embeddings
        embeds = self.generation_encoder.codebook(tokens)
        
        # Reshape to spatial dimensions
        h = w = int(math.sqrt(tokens.shape[1]))
        embeds = embeds.view(batch_size, h, w, -1).permute(0, 3, 1, 2)
        
        # Decode
        return self.generation_encoder.decoder(embeds)


class SingleEncoderBaseline(nn.Module):
    """
    Baseline: Single Visual Encoder (Chameleon-style)
    Uses a single encoder for both understanding and generation.
    """
    
    def __init__(self, config: FrameworkConfig):
        super().__init__()
        self.config = config
        
        # Single visual encoder (VQGAN-style)
        self.visual_encoder = VisualGenerationEncoder(config)
        
        # Text embedding
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Projection
        self.projector = nn.Sequential(
            nn.Linear(config.generation_encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Transformer backbone
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.feedforward_dim,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Output heads
        self.understanding_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.generation_head = nn.Linear(config.hidden_dim, config.codebook_size)
    
    def forward(
        self,
        image: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        task_type: str = "understanding"
    ) -> Dict[str, torch.Tensor]:
        # Encode image
        _, visual_features, vq_loss = self.visual_encoder(image)
        
        # visual_features is already (B, seq_len, hidden_dim) from visual_encoder
        # No additional projection needed
        
        # Combine with text if provided
        if text_tokens is not None:
            text_embeds = self.text_embed(text_tokens[:, :-1])
            combined = torch.cat([
                visual_features.mean(dim=1, keepdim=True).expand(-1, text_embeds.size(1), -1),
                text_embeds
            ], dim=1)
        else:
            combined = visual_features
        
        # Forward through transformer
        for layer in self.layers:
            combined = layer(combined)
        
        combined = self.norm(combined)
        
        # Output heads
        understanding_logits = self.understanding_head(combined)
        generation_logits = self.generation_head(combined)
        
        return {
            "understanding_logits": understanding_logits,
            "generation_logits": generation_logits,
            "vq_loss": vq_loss
        }


def create_model(model_type: str = "unified", config: Optional[FrameworkConfig] = None) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: "unified" for decoupled encoders, "baseline" for single encoder
        config: Model configuration
    
    Returns:
        Model instance
    """
    if config is None:
        config = FrameworkConfig()
    
    if model_type == "unified":
        return UnifiedFramework(config)
    elif model_type == "baseline":
        return SingleEncoderBaseline(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")