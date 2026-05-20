"""
U-Transformer architecture for global weather forecasting.
Combines U-Net multi-scale features with Transformer long-range dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """Multi-head self-attention for spatial tokens."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with spatial attention and MLP."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    """Double conv block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        return self.conv(x) + self.skip(x)


class UTransformer(nn.Module):
    """
    U-Transformer for weather forecasting.
    """
    def __init__(self, in_channels=70, out_channels=70, base_ch=64,
                 num_scales=4, transformer_depth=4, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.input_proj = nn.Conv2d(in_channels, 70, 1) if in_channels != 70 else nn.Identity()
        
        # Encoder
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = base_ch
        self.enc.append(ConvBlock(70, ch))
        
        for i in range(1, num_scales):
            self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            next_ch = ch * 2
            self.enc.append(ConvBlock(ch, next_ch))
            ch = next_ch
        
        # Bottleneck Transformer
        self.bottleneck = nn.ModuleList([
            TransformerBlock(ch, num_heads=num_heads)
            for _ in range(transformer_depth)
        ])
        
        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(num_scales - 1, 0, -1):
            next_ch = ch // 2
            self.up.append(nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1))
            self.dec.append(ConvBlock(ch, next_ch))  # concat skip -> 2*next_ch input, but up outputs next_ch
            ch = next_ch
        
        self.final = nn.Conv2d(ch, out_channels, 1)
        
    def forward(self, x):
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b, t * c, h, w)
        x = self.input_proj(x)
        
        # Encoder
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            skips.append(x)
            if i < len(self.down):
                x = self.down[i](x)
        
        # Bottleneck
        b, c, h, w = x.shape
        x_tok = x.view(b, c, h * w).permute(0, 2, 1)
        for tblk in self.bottleneck:
            x_tok = tblk(x_tok)
        x = x_tok.permute(0, 2, 1).view(b, c, h, w)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            skip = skips[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        
        return self.final(x)


class CascadeUTransformer(nn.Module):
    """Cascade of three specialized U-Transformer models."""
    def __init__(self, in_channels=70, out_channels=70, base_ch=64):
        super().__init__()
        self.stage1 = UTransformer(in_channels, out_channels, base_ch=base_ch,
                                   num_scales=4, transformer_depth=4, num_heads=8)
        self.stage2 = UTransformer(in_channels, out_channels, base_ch=base_ch,
                                   num_scales=4, transformer_depth=3, num_heads=8)
        self.stage3 = UTransformer(in_channels, out_channels, base_ch=base_ch,
                                   num_scales=3, transformer_depth=2, num_heads=8)
        
    def forward(self, x, stage=1):
        if stage == 1:
            return self.stage1(x)
        elif stage == 2:
            return self.stage2(x)
        elif stage == 3:
            return self.stage3(x)
        else:
            raise ValueError(f"Invalid stage {stage}")
    
    def rollout(self, x0, num_steps=60, switch_points=[20, 40]):
        forecasts = []
        x = x0
        for step in range(num_steps):
            if step < switch_points[0]:
                stage = 1
            elif step < switch_points[1]:
                stage = 2
            else:
                stage = 3
            x = self.forward(x, stage=stage)
            forecasts.append(x.detach().cpu())
        return forecasts


class SingleModelBaseline(nn.Module):
    """Single U-Transformer baseline without cascade."""
    def __init__(self, in_channels=70, out_channels=70, base_ch=64):
        super().__init__()
        self.model = UTransformer(in_channels, out_channels, base_ch=base_ch,
                                  num_scales=4, transformer_depth=4, num_heads=8)
    
    def forward(self, x):
        return self.model(x)
    
    def rollout(self, x0, num_steps=60):
        forecasts = []
        x = x0
        for _ in range(num_steps):
            x = self.forward(x)
            forecasts.append(x.detach().cpu())
        return forecasts
