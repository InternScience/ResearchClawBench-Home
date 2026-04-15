"""
Decoupled Visual Encoding Framework for Unified Multimodal Autoregressive Models

This module implements a unified autoregressive framework that decouples visual encoding
into two specialized pathways:
  1. Understanding Encoder: High-resolution, detail-preserving, optimized for OCR/VQA
  2. Generation Encoder: Reconstruction-focused, optimized for image synthesis quality

Both pathways produce discrete visual tokens consumed by a shared Transformer backbone.

References:
  - Chameleon (Meta FAIR, 2024): Early-fusion token-based mixed-modal models
  - LLaVA (Liu et al., 2023): Visual instruction tuning with projection layer
  - LlamaGen (Sun et al., 2024): Vanilla AR models for scalable image generation
  - SigLIP (Zhai et al., 2023): Sigmoid loss for language-image pre-training
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FrameworkConfig:
    """Configuration for the Decoupled Visual Encoding Framework."""
    # Shared Transformer backbone (following Llama architecture)
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    vocab_size: int = 65536  # Includes text tokens + image codebook tokens
    
    # Understanding encoder (high-res, detail-preserving)
    understanding_encoder_depth: int = 24
    understanding_downsample_ratio: int = 8  # Higher resolution
    understanding_codebook_size: int = 16384
    understanding_feature_dim: int = 8
    
    # Generation encoder (reconstruction-focused)
    generation_encoder_depth: int = 16
    generation_downsample_ratio: int = 16  # Standard resolution
    generation_codebook_size: int = 8192
    generation_feature_dim: int = 8
    
    # Training stability (from Chameleon)
    use_qk_norm: bool = True
    use_dropout: float = 0.1
    norm_placement: str = "post_attention"  # Swin-style normalization
    
    # Tokenization
    max_image_tokens: int = 1024  # 32x32 grid at ratio=16
    pad_token_id: int = 0
    img_start_token_id: int = 65000
    img_end_token_id: int = 65001


class DiscreteVisualTokenizer:
    """
    Simulates a VQGAN-based discrete visual tokenizer.
    
    Encodes images into discrete tokens using a learned codebook.
    Supports both understanding (high-res) and generation (standard-res) modes.
    """
    
    def __init__(self, config: FrameworkConfig, mode: str = "understanding"):
        self.config = config
        self.mode = mode
        
        if mode == "understanding":
            self.codebook_size = config.understanding_codebook_size
            self.downsample_ratio = config.understanding_downsample_ratio
            self.feature_dim = config.understanding_feature_dim
            self.encoder_depth = config.understanding_encoder_depth
        else:  # generation
            self.codebook_size = config.generation_codebook_size
            self.downsample_ratio = config.generation_downsample_ratio
            self.feature_dim = config.generation_feature_dim
            self.encoder_depth = config.generation_encoder_depth
        
        # Initialize codebook (simulated)
        rng = np.random.RandomState(42)
        self.codebook = rng.randn(self.codebook_size, self.feature_dim).astype(np.float32)
        self.codebook = self.codebook / np.linalg.norm(self.codebook, axis=1, keepdims=True)
        
        # Simulated encoder/decoder weights
        self.encoder_proj = rng.randn(3, self.feature_dim).astype(np.float32) * 0.1
        self.decoder_proj = rng.randn(self.feature_dim, 3).astype(np.float32) * 0.1
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode an image into discrete tokens.
        
        Args:
            image: Input image of shape (H, W, 3) with values in [0, 255]
        
        Returns:
            token_grid: Discrete token indices of shape (h, w)
        """
        h, w = image.shape[:2]
        token_h, token_w = h // self.downsample_ratio, w // self.downsample_ratio
        
        # Simulate encoding: downsample and quantize
        downsampled = self._downsample(image, token_h, token_w)
        
        # Project to feature space
        features = np.einsum('hwc,cd->hwd', downsampled / 255.0, self.encoder_proj)
        
        # Quantize: find nearest codebook entry for each spatial position
        features_flat = features.reshape(-1, self.feature_dim)
        codebook_norm = self.codebook / np.linalg.norm(self.codebook, axis=1, keepdims=True)
        similarities = features_flat @ codebook_norm.T
        token_indices = np.argmax(similarities, axis=1)
        
        return token_indices.reshape(token_h, token_w)
    
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """
        Decode discrete tokens back to image pixels.
        
        Args:
            tokens: Token indices of shape (h, w)
        
        Returns:
            image: Reconstructed image of shape (H, W, 3)
        """
        # Lookup codebook vectors
        h, w = tokens.shape
        features = self.codebook[tokens.flatten()].reshape(h, w, self.feature_dim)
        
        # Project to RGB
        reconstructed = np.einsum('hwd,dc->hwc', features, self.decoder_proj)
        reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
        
        # Upsample to original resolution
        target_h, target_w = h * self.downsample_ratio, w * self.downsample_ratio
        upsampled = self._upsample(reconstructed, target_h, target_w)
        
        return upsampled
    
    def _downsample(self, image: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Simple average pooling downsampling."""
        h, w = image.shape[:2]
        result = np.zeros((th, tw, 3), dtype=np.float32)
        for i in range(th):
            for j in range(tw):
                y_start, y_end = i * (h // th), (i + 1) * (h // th)
                x_start, x_end = j * (w // tw), (j + 1) * (w // tw)
                result[i, j] = image[y_start:y_end, x_start:x_end].mean(axis=(0, 1))
        return result
    
    def _upsample(self, image: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Nearest-neighbor upsampling."""
        h, w = image.shape[:2]
        result = np.zeros((th, tw, 3), dtype=np.uint8)
        scale_y, scale_x = th / h, tw / w
        for i in range(th):
            for j in range(tw):
                src_i, src_j = int(i / scale_y), int(j / scale_x)
                result[i, j] = image[min(src_i, h-1), min(src_j, w-1)]
        return result
    
    @property
    def num_tokens(self) -> int:
        """Total number of tokens for a standard 512x512 image."""
        return (512 // self.downsample_ratio) ** 2


class TaskAdaptiveRouter:
    """
    Routes visual tokens through the appropriate pathway based on task type.
    
    Supports dynamic switching between understanding and generation modes
    within a single forward pass.
    """
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Routing projection layers
        rng = np.random.RandomState(123)
        self.understanding_proj = rng.randn(config.hidden_size, config.hidden_size).astype(np.float32) * 0.02
        self.generation_proj = rng.randn(config.hidden_size, config.hidden_size).astype(np.float32) * 0.02
        
        # Gating mechanism for soft routing
        self.gate_proj = rng.randn(config.hidden_size, 2).astype(np.float32) * 0.01
    
    def route(self, hidden_states: np.ndarray, task_type: str = "understanding") -> np.ndarray:
        """
        Route hidden states through the appropriate pathway.
        
        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, hidden_size)
            task_type: One of "understanding", "generation", or "mixed"
        
        Returns:
            Routed hidden states
        """
        if task_type == "understanding":
            return hidden_states @ self.understanding_proj
        elif task_type == "generation":
            return hidden_states @ self.generation_proj
        else:  # mixed: use soft gating
            gate_logits = hidden_states @ self.gate_proj
            gate_weights = self._softmax(gate_logits, axis=-1)
            routed = (hidden_states @ self.understanding_proj) * gate_weights[..., 0:1] + \
                     (hidden_states @ self.generation_proj) * gate_weights[..., 1:2]
            return routed
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SharedAutoregressiveTransformer:
    """
    Shared Transformer backbone for both understanding and generation tasks.
    
    Follows the Llama architecture with stability modifications from Chameleon:
    - RMSNorm normalization
    - SwiGLU activation
    - Query-Key normalization for stability
    - Post-attention normalization placement
    """
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        rng = np.random.RandomState(456)
        
        # Token embeddings (shared vocabulary)
        self.token_embeddings = rng.randn(config.vocab_size, config.hidden_size).astype(np.float32) * 0.02
        self.token_embeddings /= np.sqrt(config.hidden_size)
        
        # Layer parameters (simplified for simulation)
        self.attn_projs = []
        self.ffn_projs = []
        for _ in range(config.num_layers):
            # Attention projections
            qkv = rng.randn(config.hidden_size, 3 * config.hidden_size).astype(np.float32) * 0.02
            o_proj = rng.randn(config.hidden_size, config.hidden_size).astype(np.float32) * 0.02
            self.attn_projs.append((qkv, o_proj))
            
            # FFN projections (SwiGLU)
            up_proj = rng.randn(config.hidden_size, config.intermediate_size).astype(np.float32) * 0.02
            gate_proj = rng.randn(config.hidden_size, config.intermediate_size).astype(np.float32) * 0.02
            down_proj = rng.randn(config.intermediate_size, config.hidden_size).astype(np.float32) * 0.02
            self.ffn_projs.append((up_proj, gate_proj, down_proj))
        
        # Final layer norm
        self.final_norm_weight = np.ones(config.hidden_size, dtype=np.float32)
        self.final_norm_bias = np.zeros(config.hidden_size, dtype=np.float32)
    
    def forward(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through the shared transformer.
        
        Args:
            input_ids: Token indices of shape (batch, seq_len)
            attention_mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden = self.token_embeddings[input_ids]
        
        # Process through transformer layers
        for layer_idx in range(self.num_layers):
            hidden = self._transformer_layer(hidden, layer_idx, attention_mask)
        
        # Final normalization and projection to vocabulary
        hidden = self._rms_norm(hidden, self.final_norm_weight, self.final_norm_bias)
        logits = hidden @ self.token_embeddings.T
        
        return logits
    
    def _transformer_layer(self, hidden: np.ndarray, layer_idx: int, 
                           attention_mask: Optional[np.ndarray]) -> np.ndarray:
        """Single transformer layer with Chameleon-style stability modifications."""
        # Pre-attention RMSNorm
        normed = self._rms_norm(hidden)
        
        # Self-attention with QK-Norm
        qkv_proj, o_proj = self.attn_projs[layer_idx]
        qkv = normed @ qkv_proj
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # QK-Normalization (Chameleon stability technique)
        q = self._rms_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self._rms_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask * -1e9
        attn_weights = self._softmax(scores, axis=-1)
        attn_output = attn_weights @ v
        
        # Output projection
        attn_output = attn_output @ o_proj
        
        # Residual connection (post-attention norm placement)
        hidden = hidden + attn_output
        
        # Feed-forward with SwiGLU
        normed = self._rms_norm(hidden)
        up_proj, gate_proj, down_proj = self.ffn_projs[layer_idx]
        
        up_state = normed @ up_proj
        gate_state = normed @ gate_proj
        gate_state = gate_state * self._sigmoid(gate_state)  # SiLU/Swish
        
        ff_output = (up_state * gate_state) @ down_proj
        
        # Dropout simulation
        if self.config.use_dropout > 0:
            ff_output *= (1 - self.config.use_dropout)
        
        # Residual connection
        hidden = hidden + ff_output
        
        return hidden
    
    @staticmethod
    def _rms_norm(x: np.ndarray, weight: Optional[np.ndarray] = None, 
                  bias: Optional[np.ndarray] = None) -> np.ndarray:
        """RMS Normalization."""
        if weight is None:
            weight = np.ones(x.shape[-1], dtype=np.float32)
        if bias is None:
            bias = np.zeros(x.shape[-1], dtype=np.float32)
        
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        normalized = x / rms
        return normalized * weight + bias
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class DecoupledMultimodalFramework:
    """
    Main framework class that integrates all components.
    
    Architecture:
        Input Image --> [Understanding Encoder] --> Discrete Tokens --+
        Input Image --> [Generation Encoder]   --> Discrete Tokens --+--> Shared Transformer --> Output
        Input Text  --------------------------------------------------+
    
    The framework supports:
    - Visual Question Answering (VQA)
    - Optical Character Recognition (OCR)
    - Text-to-Image Generation
    - Mixed-modal interleaved generation
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        
        # Initialize components
        self.understanding_tokenizer = DiscreteVisualTokenizer(self.config, mode="understanding")
        self.generation_tokenizer = DiscreteVisualTokenizer(self.config, mode="generation")
        self.router = TaskAdaptiveRouter(self.config)
        self.transformer = SharedAutoregressiveTransformer(self.config)
    
    def understand_image(self, image: np.ndarray, question: str = "") -> Dict:
        """
        Process an image for understanding tasks (VQA, OCR, etc.).
        
        Uses the high-resolution understanding encoder pathway.
        """
        # Encode with understanding tokenizer
        tokens = self.understanding_tokenizer.encode(image)
        token_seq = tokens.flatten()
        
        # Build input sequence: [IMG_START] + image_tokens + [IMG_END] + question_tokens
        # (Simplified: we use token IDs directly)
        input_ids = np.array([
            [self.config.img_start_token_id] + 
            list(token_seq[:min(len(token_seq), self.config.max_image_tokens)]) +
            [self.config.img_end_token_id]
        ], dtype=np.int64)
        
        # Forward through transformer
        logits = self.transformer.forward(input_ids)
        
        return {
            "tokens": tokens,
            "num_tokens": len(token_seq),
            "logits_shape": logits.shape,
            "task_type": "understanding",
            "encoder_mode": "high_resolution",
            "downsample_ratio": self.config.understanding_downsample_ratio,
            "codebook_size": self.config.understanding_codebook_size
        }
    
    def generate_image(self, text_prompt: str, condition_tokens: Optional[np.ndarray] = None) -> Dict:
        """
        Generate an image from text prompt.
        
        Uses the generation encoder pathway for optimal reconstruction quality.
        """
        # Simulate autoregressive generation
        gen_h = 512 // self.config.generation_downsample_ratio
        gen_w = 512 // self.config.generation_downsample_ratio
        
        # Generate tokens autoregressively (simulated)
        rng = np.random.RandomState(789)
        generated_tokens = rng.randint(0, self.config.generation_codebook_size, 
                                        size=(gen_h, gen_w), dtype=np.int32)
        
        # Decode to image
        generated_image = self.generation_tokenizer.decode(generated_tokens)
        
        return {
            "generated_tokens": generated_tokens,
            "generated_image": generated_image,
            "num_tokens": gen_h * gen_w,
            "task_type": "generation",
            "encoder_mode": "reconstruction_optimized",
            "downsample_ratio": self.config.generation_downsample_ratio,
            "codebook_size": self.config.generation_codebook_size
        }
    
    def mixed_modal_forward(self, image: np.ndarray, text_input: str, 
                            task_type: str = "understanding") -> Dict:
        """
        Unified forward pass supporting both understanding and generation.
        """
        if task_type == "understanding":
            return self.understand_image(image, text_input)
        else:
            return self.generate_image(text_input)
    
    def get_architecture_summary(self) -> Dict:
        """Return a summary of the framework architecture."""
        total_params = self._estimate_parameters()
        
        return {
            "framework_name": "DecoupledVisualEncoding-AR",
            "total_parameters_millions": round(total_params / 1e6, 2),
            "shared_transformer": {
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "vocab_size": self.config.vocab_size
            },
            "understanding_encoder": {
                "depth": self.config.understanding_encoder_depth,
                "downsample_ratio": self.config.understanding_downsample_ratio,
                "codebook_size": self.config.understanding_codebook_size,
                "output_tokens_per_image": (512 // self.config.understanding_downsample_ratio) ** 2
            },
            "generation_encoder": {
                "depth": self.config.generation_encoder_depth,
                "downsample_ratio": self.config.generation_downsample_ratio,
                "codebook_size": self.config.generation_codebook_size,
                "output_tokens_per_image": (512 // self.config.generation_downsample_ratio) ** 2
            },
            "stability_techniques": {
                "qk_normalization": self.config.use_qk_norm,
                "dropout_rate": self.config.use_dropout,
                "norm_placement": self.config.norm_placement
            }
        }
    
    def _estimate_parameters(self) -> int:
        """Estimate total parameter count."""
        # Embedding layer
        params = self.config.vocab_size * self.config.hidden_size
        
        # Transformer layers
        per_layer = (
            4 * self.config.hidden_size ** 2 +  # QKV + O projection
            3 * self.config.hidden_size * self.config.intermediate_size +  # SwiGLU FFN
            4 * self.config.hidden_size  # Layer norms
        )
        params += per_layer * self.config.num_layers
        
        # Router projections
        params += 3 * self.config.hidden_size ** 2
        
        # Tokenizer codebooks
        params += (self.config.understanding_codebook_size + 
                   self.config.generation_codebook_size) * self.config.understanding_feature_dim
        
        return params


def run_framework_demo():
    """Run a demonstration of the framework with the provided data files."""
    import os
    from PIL import Image
    
    config = FrameworkConfig()
    framework = DecoupledMultimodalFramework(config)
    
    results = {}
    
    # Load and process equation.png (OCR task)
    equation_path = os.path.join(os.path.dirname(__file__), "..", "data", "equation.png")
    if os.path.exists(equation_path):
        equation_img = np.array(Image.open(equation_path).convert("RGB"))
        eq_result = framework.understand_image(equation_img, "What is this equation?")
        results["equation_ocr"] = eq_result
    
    # Load and process doge.png (semantic understanding task)
    doge_path = os.path.join(os.path.dirname(__file__), "..", "data", "doge.png")
    if os.path.exists(doge_path):
        doge_img = np.array(Image.open(doge_path).convert("RGB"))
        doge_result = framework.understand_image(doge_img, "Describe this meme")
        results["doge_understanding"] = doge_result
    
    # Simulate text-to-image generation
    gen_result = framework.generate_image("A majestic mountain landscape at sunset")
    results["text_to_image"] = gen_result
    
    # Architecture summary
    results["architecture"] = framework.get_architecture_summary()
    
    return results


if __name__ == "__main__":
    results = run_framework_demo()
    print("Framework demo completed successfully.")
    print(f"Architecture: {results['architecture']['framework_name']}")
    print(f"Total parameters: {results['architecture']['total_parameters_millions']}M")
