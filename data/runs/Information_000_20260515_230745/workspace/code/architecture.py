"""
Decoupled Visual Encoding (DVE) for Unified Multimodal Autoregressive Models
=======================================================================

Architecture design: A single Transformer that handles both multimodal
understanding and visual generation by decoupling visual encoding into:
1. Understanding Pathway: dense continuous features from a vision encoder
2. Generation Pathway: discrete tokens from an image tokenizer (VQGAN)

The key innovation is that these two pathways share a common visual backbone
but branch at the encoding stage, allowing the model to use the most
appropriate representation for each task.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DVEConfig:
    """Configuration for Decoupled Visual Encoding architecture."""
    # Transformer backbone
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 65536

    # Vision encoder (shared backbone)
    image_size: int = 512
    patch_size: int = 16
    vision_hidden_dim: int = 1024
    vision_num_layers: int = 24

    # Understanding pathway (continuous features)
    proj_dim: int = 4096  # projection to LLM embedding space

    # Generation pathway (discrete tokens)
    codebook_size: int = 16384
    codebook_dim: int = 8
    downsample_ratio: int = 16  # produces 32x32 = 1024 tokens for 512x512

    # Decoupling factor: controls degree of shared vs. separate processing
    shared_layers: int = 12  # first N vision layers are shared
    understand_layers: int = 6  # layers for understanding branch
    generate_layers: int = 6   # layers for generation branch


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.scale = np.ones(dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


class RotaryPositionalEmbedding:
    """2D Rotary Positional Embedding for vision features."""

    def __init__(self, dim: int, max_height: int = 64, max_width: int = 64):
        self.dim = dim
        theta = 10000.0 ** (-2.0 * np.arange(0, dim, 2) / dim)
        self.theta = theta

    def compute_freqs(self, h: int, w: int) -> np.ndarray:
        y_freqs = np.outer(np.arange(h), self.theta[:self.dim // 4])
        x_freqs = np.outer(np.arange(w), self.theta[:self.dim // 4])
        return y_freqs, x_freqs


class VisionEncoder:
    """Shared vision encoder backbone (ViT-style)."""

    def __init__(self, config: DVEConfig):
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_dim = config.vision_hidden_dim
        self.num_patches = (config.image_size // config.patch_size) ** 2

    def patchify(self, image: np.ndarray) -> np.ndarray:
        """Convert image to patches. [C, H, W] -> [N, P^2*C]"""
        C, H, W = image.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0
        patches = image.reshape(C, H // p, p, W // p, p)
        patches = patches.transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(-1, p * p * C)
        return patches

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Extract patch embeddings."""
        patches = self.patchify(image)
        return patches


class UnderstandingPathway:
    """Continuous feature pathway for multimodal understanding.

    Projects dense visual features into the LLM embedding space.
    Similar to LLaVA's approach: vision encoder + projector -> LLM.
    """

    def __init__(self, config: DVEConfig):
        self.config = config
        self.vision_dim = config.vision_hidden_dim
        self.llm_dim = config.proj_dim

    def project(self, vision_features: np.ndarray) -> np.ndarray:
        """Project vision features to LLM embedding space.

        Uses a two-layer MLP with GELU activation (stronger than linear).
        """
        num_tokens, feat_dim = vision_features.shape

        # Simulate MLP projection
        hidden = vision_features @ np.random.randn(feat_dim, self.llm_dim) * 0.02
        hidden = np.maximum(0, hidden)  # ReLU
        projected = hidden @ np.random.randn(self.llm_dim, self.llm_dim) * 0.02
        return projected


class GenerationPathway:
    """Discrete token pathway for visual generation.

    Uses VQGAN-style quantization to convert continuous features
    into discrete tokens for autoregressive generation.
    """

    def __init__(self, config: DVEConfig):
        self.config = config
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        # Initialize codebook
        self.codebook = np.random.randn(self.codebook_size, self.codebook_dim) * 0.02
        self.codebook = self.codebook / np.linalg.norm(
            self.codebook, axis=1, keepdims=True
        )

    def quantize(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize continuous features to discrete tokens.

        Returns:
            quantized: Quantized feature vectors
            indices: Codebook indices
        """
        # Normalize features
        feat_norm = features / (
            np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
        )

        # Find nearest codebook vector
        similarities = feat_norm @ self.codebook.T
        indices = np.argmax(similarities, axis=-1)

        # Lookup quantized vectors
        quantized = self.codebook[indices]
        return quantized, indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Convert token indices back to feature vectors."""
        return self.codebook[indices]


class DecoupledVisualEncoder:
    """Main DVE module: decouples visual encoding into two pathways.

    Architecture:
    ┌─────────────────────────────────────┐
    │         Input Image                  │
    └─────────────────┬───────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │    Shared Vision Backbone (ViT)      │
    │    Layers 1..shared_layers           │
    └────────┬──────────────────┬─────────┘
             │                  │
    ┌────────▼────────┐  ┌──────▼──────────┐
    │ Understanding   │  │ Generation       │
    │ Pathway         │  │ Pathway          │
    │ (Continuous)    │  │ (Discrete Tokens) │
    │ MLP Projector   │  │ VQGAN Quantizer  │
    └────────┬────────┘  └──────┬──────────┘
             │                  │
    ┌────────▼──────────────────▼──────────┐
    │    Unified Transformer Decoder        │
    │    (Autoregressive next-token pred.)   │
    └──────────────────────────────────────┘
    """

    def __init__(self, config: DVEConfig):
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.understand_path = UnderstandingPathway(config)
        self.generate_path = GenerationPathway(config)

    def encode_for_understanding(
        self, image: np.ndarray
    ) -> np.ndarray:
        """Encode image for multimodal understanding tasks."""
        features = self.vision_encoder.encode(image)
        return self.understand_path.project(features)

    def encode_for_generation(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode image for visual generation tasks."""
        features = self.vision_encoder.encode(image)
        # Reduce to codebook dimension
        reduced = features[:, :self.config.codebook_dim]
        return self.generate_path.quantize(reduced)

    def compute_encoding_efficiency(
        self, image: np.ndarray
    ) -> dict:
        """Compute efficiency metrics for both pathways."""
        h, w = image.shape[1], image.shape[2]
        num_patches = (h // self.config.patch_size) * (w // self.config.patch_size)

        understand_tokens = num_patches
        generate_tokens = (
            h // self.config.downsample_ratio
        ) * (w // self.config.downsample_ratio)

        return {
            "image_size": f"{h}x{w}",
            "num_patches": num_patches,
            "understand_tokens": understand_tokens,
            "generate_tokens": generate_tokens,
            "understand_dim": self.config.proj_dim,
            "generate_dim": self.config.codebook_dim,
            "understand_total_bytes": understand_tokens * self.config.proj_dim * 2,
            "generate_total_bytes": generate_tokens * self.config.codebook_dim,
        }
