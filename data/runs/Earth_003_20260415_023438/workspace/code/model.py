"""
Cascade U-Transformer Weather Forecasting System

Architecture: Three-stage cascade of U-Transformer models
- Stage 1 (Short-range): 0-5 days (steps 1-20)
- Stage 2 (Medium-range): 5-10 days (steps 21-40)  
- Stage 3 (Extended-range): 10-15 days (steps 41-60)

Each U-Transformer combines:
- U-Net style encoder-decoder with skip connections
- Transformer attention in the bottleneck
- Variable-group-specific processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    """Convolutional block for U-Net encoder/decoder."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UTransformer(nn.Module):
    """
    U-Transformer: U-Net architecture with Transformer bottleneck.
    
    Combines local feature extraction via U-Net skip connections
    with global context modeling via Transformer attention.
    
    Input: Two consecutive atmospheric states concatenated along channel dim
           Shape: (B, 2*C, H, W) where C=70 variables
    Output: Next atmospheric state
            Shape: (B, C, H, W)
    """
    def __init__(self, in_channels=140, out_channels=70, base_dim=64, 
                 num_transformer_blocks=4, num_heads=4, 
                 input_h=181, input_w=360):
        super().__init__()
        
        # Encoder path
        self.enc1 = ConvBlock(in_channels, base_dim)
        self.enc2 = ConvBlock(base_dim, base_dim * 2)
        self.enc3 = ConvBlock(base_dim * 2, base_dim * 4)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck with Transformer
        self.bottleneck_conv = ConvBlock(base_dim * 4, base_dim * 8)
        
        # Calculate bottleneck spatial dimensions
        self.bottleneck_h = input_h // 8
        self.bottleneck_w = input_w // 8
        bottleneck_dim = base_dim * 8
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.bottleneck_h * self.bottleneck_w, bottleneck_dim) * 0.02
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(bottleneck_dim, num_heads, mlp_ratio=2, dropout=0.1)
            for _ in range(num_transformer_blocks)
        ])
        self.norm = nn.LayerNorm(bottleneck_dim)
        
        # Decoder path
        self.up3 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_dim * 8, base_dim * 4)
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_dim * 4, base_dim * 2)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.dec1 = ConvBlock(base_dim * 2, base_dim)
        
        # Output projection
        self.out_conv = nn.Conv2d(base_dim, out_channels, 1)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck_conv(self.pool(e3))
        
        # Transformer processing
        b_flat = b.flatten(2).transpose(1, 2)  # B, H*W, C
        b_flat = b_flat + self.pos_embed
        for block in self.transformer_blocks:
            b_flat = block(b_flat)
        b_flat = self.norm(b_flat)
        b = b_flat.transpose(1, 2).reshape(B, -1, self.bottleneck_h, self.bottleneck_w)
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out = self.out_conv(d1)
        return out


class CascadeUTransformer(nn.Module):
    """
    Three-stage cascade U-Transformer system.
    
    Each stage specializes in a different forecast range:
    - Stage 1: Short-range (0-5 days, steps 1-20)
    - Stage 2: Medium-range (5-10 days, steps 21-40)
    - Stage 3: Extended-range (10-15 days, steps 41-60)
    
    The cascade mitigates error accumulation by having each stage
    trained/optimized for its specific lead-time range.
    
    Input: Two consecutive atmospheric states concatenated
           Shape: (B, 140, H, W) - 70 variables × 2 time steps
    Output: 60 forecast steps, each (B, 70, H, W)
    """
    def __init__(self, in_channels=140, out_channels=70, base_dim=64,
                 num_transformer_blocks=4, num_heads=4,
                 input_h=181, input_w=360):
        super().__init__()
        
        self.out_channels = out_channels
        
        self.stage1 = UTransformer(
            in_channels, out_channels, base_dim,
            num_transformer_blocks, num_heads, input_h, input_w
        )
        self.stage2 = UTransformer(
            in_channels, out_channels, base_dim,
            num_transformer_blocks, num_heads, input_h, input_w
        )
        self.stage3 = UTransformer(
            in_channels, out_channels, base_dim,
            num_transformer_blocks, num_heads, input_h, input_w
        )
        
        # Stage transition layers for error correction
        # These take the concatenation of the new prediction and the 
        # previous stage's prediction to produce a corrected output
        self.transition1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.GELU(),
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.GELU(),
        )
        
    def make_input(self, prev_state, current_state):
        """Create model input from two consecutive states."""
        return torch.cat([prev_state, current_state], dim=1)
    
    def forward(self, x, max_steps=60):
        """
        Autoregressive forecast for max_steps 6-hour intervals.
        
        x: initial input (B, 140, H, W) - two time steps concatenated
        Returns: list of 60 forecast steps, each (B, 70, H, W)
        
        Stage 1 handles steps 1-20 (0-5 days)
        Stage 2 handles steps 21-40 (5-10 days)
        Stage 3 handles steps 41-60 (10-15 days)
        """
        forecasts = []
        
        # Initial two states
        state_t0 = x[:, :self.out_channels]
        state_t1 = x[:, self.out_channels:]
        prev = state_t0
        current = state_t1
        
        # Stage 1: Short-range (steps 1-20)
        for step in range(20):
            inp = self.make_input(prev, current)
            next_state = self.stage1(inp)
            forecasts.append(next_state)
            prev = current
            current = next_state
        
        # Transition to Stage 2 with error correction
        stage1_last = forecasts[-1]
        inp = self.make_input(prev, current)
        stage2_first = self.stage2(inp)
        corrected = self.transition1(torch.cat([stage2_first, stage1_last], dim=1))
        forecasts.append(corrected)
        prev = current
        current = corrected
        
        # Stage 2: Medium-range (steps 21-40, already did step 21)
        for step in range(19):
            inp = self.make_input(prev, current)
            next_state = self.stage2(inp)
            forecasts.append(next_state)
            prev = current
            current = next_state
        
        # Transition to Stage 3 with error correction
        stage2_last = forecasts[-1]
        inp = self.make_input(prev, current)
        stage3_first = self.stage3(inp)
        corrected = self.transition2(torch.cat([stage3_first, stage2_last], dim=1))
        forecasts.append(corrected)
        prev = current
        current = corrected
        
        # Stage 3: Extended-range (steps 41-60, already did step 41)
        for step in range(19):
            inp = self.make_input(prev, current)
            next_state = self.stage3(inp)
            forecasts.append(next_state)
            prev = current
            current = next_state
        
        return forecasts


if __name__ == "__main__":
    # Test with small input
    H, W = 45, 90
    model = CascadeUTransformer(
        in_channels=140, out_channels=70, base_dim=32,
        num_transformer_blocks=2, num_heads=4,
        input_h=H, input_w=W
    )
    x = torch.randn(1, 140, H, W)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    out = model.stage1(x)
    print(f"Stage 1 output shape: {out.shape}")
