#!/usr/bin/env python3
"""Cascade U-Transformer Model for 15-Day Weather Forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm2(x + self.attn(self.norm1(x))))


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        return self.pool(x), x


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x, skip], dim=1))


class UTransformer(nn.Module):
    def __init__(self, in_channels: int = 140, out_channels: int = 70, base_channels: int = 64,
                 embed_dim: int = 256, num_heads: int = 8, num_blocks: int = 4, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        
        self.init_conv = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))
        
        self.enc1 = DownsampleBlock(base_channels, base_channels)
        self.enc2 = DownsampleBlock(base_channels, base_channels*2)
        self.enc3 = DownsampleBlock(base_channels*2, base_channels*4)
        self.enc4 = DownsampleBlock(base_channels*4, base_channels*8)
        
        bot_ch = base_channels * 8
        self.bot_embed = PatchEmbedding(bot_ch, embed_dim, patch_size)
        self.bot_trans = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        self.bot_proj = nn.Conv2d(embed_dim, bot_ch, 1)
        
        self.dec4 = UpsampleBlock(bot_ch, base_channels*8, base_channels*4)
        self.dec3 = UpsampleBlock(base_channels*4, base_channels*4, base_channels*2)
        self.dec2 = UpsampleBlock(base_channels*2, base_channels*2, base_channels)
        self.dec1 = UpsampleBlock(base_channels, base_channels, base_channels)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 3, padding=1), nn.BatchNorm2d(base_channels//2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        
        B, _, h, w = x4.shape
        ps = self.patch_size
        tokens = self.bot_embed(x4)
        tokens = self.bot_trans(tokens)
        spatial = tokens[:, :h//ps*w//ps].reshape(B, h//ps, w//ps, -1).permute(0,3,1,2)
        x4 = self.bot_proj(spatial)
        
        x4 = self.dec4(x4, s4)
        x3 = self.dec3(x4, s3)
        x2 = self.dec2(x3, s2)
        x1 = self.dec1(x2, s1)
        return self.out_conv(x1)


class CascadeForecaster(nn.Module):
    def __init__(self, in_channels: int = 70, out_channels: int = 70):
        super().__init__()
        configs = [
            {'base_channels': 64, 'embed_dim': 256, 'num_heads': 8, 'num_blocks': 4, 'patch_size': 2},
            {'base_channels': 48, 'embed_dim': 192, 'num_heads': 6, 'num_blocks': 3, 'patch_size': 4},
            {'base_channels': 32, 'embed_dim': 128, 'num_heads': 4, 'num_blocks': 2, 'patch_size': 8}
        ]
        self.models = nn.ModuleList([UTransformer(in_channels*2, out_channels, **c) for c in configs])
        self.configs = configs
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor, steps: int = 60, all_steps: bool = True) -> torch.Tensor:
        B, C_tot, H, W = x.shape
        C = self.out_channels
        
        if C_tot == 2*C:
            x_prev, x_curr = x[:,:C], x[:,C:]
        else:
            x_prev = x_curr = x
        
        forecasts = []
        for step in range(steps):
            hr = (step+1)*6
            midx = 0 if hr <= 72 else (1 if hr <= 168 else 2)
            inp = torch.cat([x_prev, x_curr], dim=1)
            nxt = self.models[midx](inp)
            nxt = 0.1*nxt + 0.9*x_curr
            if all_steps: forecasts.append(nxt)
            x_prev, x_curr = x_curr, nxt
        
        return torch.stack(forecasts, dim=1) if all_steps else x_curr
    
    def summary(self) -> str:
        s = "Cascade U-Transformer\n" + "="*40 + "\n"
        names = ["Short (0-3d)", "Medium (3-7d)", "Extended (7-15d)"]
        for i, (c, n) in enumerate(zip(self.configs, names)):
            s += f"M{i+1} ({n}): ch={c['base_channels']}, emb={c['embed_dim']}, heads={c['num_heads']}, blocks={c['num_blocks']}, ps={c['patch_size']}\n"
        s += f"Total params: {sum(p.numel() for p in self.parameters()):,}\n"
        return s


def test():
    print("Testing Cascade U-Transformer...")
    print("="*60)
    model = CascadeForecaster(70, 70).to(DEVICE)
    print(model.summary())
    
    x = torch.randn(1, 140, 64, 128).to(DEVICE)
    print(f"Input: {x.shape}")
    
    with torch.no_grad():
        out = model(x, steps=4, all_steps=True)
    print(f"Output (4 steps): {out.shape}")
    assert out.shape == (1, 4, 70, 64, 128)
    print("✓ Test passed!")
    return model


if __name__ == '__main__':
    test()
    with open('outputs/model_architecture.json', 'w') as f:
        json.dump({'model': 'Cascade U-Transformer', 'in_ch': 70, 'out_ch': 70, 'models': 3, 'horizon_days': 15, 'device': str(DEVICE)}, f, indent=2)
    print("Saved outputs/model_architecture.json")
