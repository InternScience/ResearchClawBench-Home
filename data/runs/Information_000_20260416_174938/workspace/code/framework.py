"""
Unified Autoregressive Framework with Decoupled Visual Encoding

This module implements a unified Transformer architecture that decouples visual encoding
for multimodal understanding and generation tasks.

Architecture Overview:
- Understanding Encoder: CLIP-style ViT for perception tasks (VQA, captioning)
- Generation Encoder: VQ-VAE tokenizer for autoregressive image generation
- Backbone: Llama-style Transformer with early fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, LlamaConfig, LlamaModel
from typing import Optional, Tuple, Dict, Any
import numpy as np


class VQVAEConfig:
    """Configuration for VQ-VAE image tokenizer."""
    def __init__(
        self,
        embedding_dim: int = 256,
        n_embeddings: int = 8192,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        downsample_ratio: int = 16,
    ):
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.downsample_ratio = downsample_ratio


class VectorQuantizer(nn.Module):
    """Vector quantization module for VQ-VAE."""
    
    def __init__(self, embedding_dim: int, n_embeddings: int, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook embedding
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)
        
        # EMA tracking
        self.register_buffer("ema_w", torch.zeros(n_embeddings))
        self.register_buffer("ema_count", torch.ones(n_embeddings) * epsilon)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            z: Continuous features [B, H, W, D]
        
        Returns:
            z_q: Quantized features
            indices: Codebook indices
            info: Quantization statistics
        """
        # Flatten spatial dimensions
        z_flat = z.reshape(-1, self.embedding_dim)
        
        # Compute distances to codebook entries
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.embedding.weight.T)
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        
        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        z_q = F.embedding(indices, self.embedding.weight).reshape(z.shape)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Compute commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        
        info = {
            "indices": indices,
            "commitment_loss": commitment_loss,
            "codebook_usage": (self.ema_count > self.epsilon).float().mean(),
        }
        
        return z_q, indices, info
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to continuous features."""
        return F.embedding(indices, self.embedding.weight)


class Encoder(nn.Module):
    """Convolutional encoder for VQ-VAE."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 256, 
                 downsample_ratio: int = 16, hidden_channels: int = 128):
        super().__init__()
        
        # Build downsampling layers
        n_down = int(np.log2(downsample_ratio))
        blocks = []
        
        blocks.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        
        for i in range(n_down):
            in_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            out_ch = min(hidden_channels * (2 ** (i + 1)), out_channels)
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            blocks.append(nn.ReLU())
        
        blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).permute(0, 2, 3, 1)  # [B, H, W, D]


class Decoder(nn.Module):
    """Convolutional decoder for VQ-VAE."""
    
    def __init__(self, in_channels: int = 256, out_channels: int = 3,
                 downsample_ratio: int = 16, hidden_channels: int = 128):
        super().__init__()
        
        n_down = int(np.log2(downsample_ratio))
        blocks = []
        
        blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        blocks.append(nn.ReLU())
        
        for i in range(n_down - 1, -1, -1):
            in_ch = min(hidden_channels * (2 ** (i + 1)), in_channels)
            out_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            blocks.append(nn.ReLU())
        
        blocks.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.permute(0, 3, 1, 2)  # [B, D, H, W]
        return torch.sigmoid(self.model(z))


class VQVAE(nn.Module):
    """Complete VQ-VAE model for image tokenization."""
    
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            out_channels=config.embedding_dim,
            downsample_ratio=config.downsample_ratio
        )
        self.quantizer = VectorQuantizer(
            embedding_dim=config.embedding_dim,
            n_embeddings=config.n_embeddings,
            decay=config.decay
        )
        self.decoder = Decoder(
            in_channels=config.embedding_dim,
            downsample_ratio=config.downsample_ratio
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Encode image to discrete tokens."""
        z = self.encoder(x)
        z_q, indices, info = self.quantizer(z)
        return indices, info
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode tokens to image."""
        z_q = self.quantizer.decode(indices)
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Full forward pass with reconstruction."""
        indices, info = self.encode(x)
        recon = self.decode(indices)
        return recon, info
    
    def get_reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss (MSE + perceptual)."""
        mse_loss = F.mse_loss(x, recon)
        return mse_loss


class DecoupledVisualEncoder(nn.Module):
    """
    Decoupled visual encoder module that routes to different encoders
    based on task type (understanding vs generation).
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        vq_config: Optional[VQVAEConfig] = None,
        projection_dim: int = 768,
    ):
        super().__init__()
        
        # Understanding encoder (CLIP ViT)
        self.understanding_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.understanding_proj = nn.Linear(
            self.understanding_encoder.config.hidden_size,
            projection_dim
        )
        
        # Generation encoder (VQ-VAE)
        if vq_config is None:
            vq_config = VQVAEConfig()
        self.generation_encoder = VQVAE(vq_config)
        
        # Task routing
        self.task_embedding = nn.Embedding(2, projection_dim)
        # 0: understanding, 1: generation
    
    def forward(
        self,
        images: torch.Tensor,
        task_type: str = "understanding"
    ) -> Dict[str, Any]:
        """
        Process images with appropriate encoder based on task.
        
        Args:
            images: Input images [B, C, H, W]
            task_type: "understanding" or "generation"
        
        Returns:
            Dictionary with task-specific outputs
        """
        if task_type == "understanding":
            # CLIP-based understanding
            clip_outputs = self.understanding_encoder(images)
            visual_features = clip_outputs.last_hidden_state  # [B, N, D]
            projected = self.understanding_proj(visual_features)
            
            return {
                "features": projected,
                "task": "understanding",
                "n_tokens": visual_features.shape[1],
            }
        
        elif task_type == "generation":
            # VQ-VAE based generation
            indices, info = self.generation_encoder.encode(images)
            
            return {
                "tokens": indices,
                "task": "generation",
                "info": info,
            }
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class UnifiedMultimodalTransformer(nn.Module):
    """
    Unified Transformer backbone for multimodal understanding and generation.
    Uses early fusion of text and visual tokens.
    """
    
    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 4096,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
        )
        
        self.backbone = LlamaModel(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Modality embeddings
        self.modality_embedding = nn.Embedding(3, hidden_size)
        # 0: text, 1: understanding_visual, 2: generation_visual
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through unified transformer.
        
        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            modality_ids: Modality type per token [B, seq_len]
            labels: Target tokens for training [B, seq_len]
        
        Returns:
            Dictionary with logits and loss
        """
        # Add modality embeddings if provided
        embeddings = self.backbone.embed_tokens(input_ids)
        
        if modality_ids is not None:
            embeddings = embeddings + self.modality_embedding(modality_ids)
        
        # Transformer forward
        outputs = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        result = {"logits": logits}
        
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss
        
        return result


def create_unified_framework(
    vocab_size: int = 65536,
    hidden_size: int = 768,
    num_layers: int = 12,
    clip_model: str = "openai/clip-vit-base-patch32",
) -> Tuple[DecoupledVisualEncoder, UnifiedMultimodalTransformer]:
    """
    Factory function to create the unified framework components.
    
    Returns:
        visual_encoder: DecoupledVisualEncoder
        transformer: UnifiedMultimodalTransformer
    """
    visual_encoder = DecoupledVisualEncoder(clip_model_name=clip_model)
    
    transformer = UnifiedMultimodalTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_layers,  # Same as layers for simplicity
        intermediate_size=hidden_size * 4,
        max_position_embeddings=4096,
    )
    
    return visual_encoder, transformer


if __name__ == "__main__":
    # Test framework creation
    print("Creating unified framework...")
    visual_encoder, transformer = create_unified_framework()
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    visual_params = sum(p.numel() for p in visual_encoder.parameters())
    
    print(f"Transformer parameters: {total_params:,}")
    print(f"Visual encoder parameters: {visual_params:,}")
    print(f"Total parameters: {total_params + visual_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    vocab_size = 65536
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    output = transformer(input_ids, attention_mask, modality_ids)
    print(f"Output logits shape: {output['logits'].shape}")
    
    print("\nFramework test completed successfully!")
