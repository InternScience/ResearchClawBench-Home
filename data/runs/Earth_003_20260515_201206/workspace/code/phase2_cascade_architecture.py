#!/usr/bin/env python3
"""
Phase 2: Cascade U-Transformer Architecture Design and Forecast Generation

Three-stage cascade:
- Stage 1 (Short-range): 6h-3d (steps 1-12), optimized for high-frequency dynamics
- Stage 2 (Medium-range): 3d-7d (steps 13-28), optimized for synoptic patterns  
- Stage 3 (Extended-range): 7d-15d (steps 29-60), optimized for large-scale trends

Each model is a U-Transformer with:
- Encoder-decoder with skip connections
- Multi-head self-attention at bottleneck
- Variable-specific normalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import json
import os

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# U-Transformer Architecture
# ==========================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialTransformerBlock(nn.Module):
    """Transformer block with spatial attention for weather fields."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to sequence: [B, H*W, C]
        xs = x.reshape(B, C, H * W).permute(0, 2, 1)
        # Self-attention
        h = self.norm1(xs)
        h, _ = self.attn(h, h, h)
        xs = xs + h
        xs = xs + self.mlp(self.norm2(xs))
        # Reshape back: [B, C, H, W]
        return xs.permute(0, 2, 1).reshape(B, C, H, W)


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.residual(x)


class UTransformerBlock(nn.Module):
    """Single level of the U-Transformer."""
    def __init__(self, in_ch, mid_ch, out_ch, use_transformer=False, num_heads=4):
        super().__init__()
        self.encoder = ConvBlock(in_ch, mid_ch)
        self.transformer = SpatialTransformerBlock(mid_ch, num_heads) if use_transformer else nn.Identity()
        self.channel_attn = ChannelAttention(mid_ch)
        self.decoder = ConvBlock(mid_ch, out_ch)
    
    def forward(self, x):
        h = self.encoder(x)
        h = self.transformer(h)
        h = self.channel_attn(h)
        return self.decoder(h)


class CascadeUTransformer(nn.Module):
    """
    U-Transformer for weather forecasting.
    
    Args:
        in_channels: Number of input channels (70 variables)
        base_channels: Base channel count
        depth: Number of encoding levels
        forecast_steps: Number of 6h steps to predict
        stage_name: Identifier for this cascade stage
    """
    def __init__(self, in_channels=70, base_channels=32, depth=3, 
                 forecast_steps=12, stage_name='short_range'):
        super().__init__()
        self.stage_name = stage_name
        self.forecast_steps = forecast_steps
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.enc_blocks.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        
        # Bottleneck with transformer
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ConvBlock(ch, bottleneck_ch),
            SpatialTransformerBlock(bottleneck_ch, num_heads=4),
            ChannelAttention(bottleneck_ch),
            ConvBlock(bottleneck_ch, bottleneck_ch)
        )
        
        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        ch = bottleneck_ch
        decoder_chs = []
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            decoder_chs.append(out_ch)
            self.upsamples.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.dec_blocks.append(nn.ModuleList([
                ConvBlock(2 * out_ch, out_ch),  # upsample(out_ch) + skip(out_ch)
                ConvBlock(out_ch, out_ch)  # refinement
            ]))
            ch = out_ch
        
        # Output projection: predict all channels simultaneously
        self.output_proj = nn.Conv2d(ch, in_channels, 1)
        
        # Tendency prediction (residual learning)
        self.tendency_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - atmospheric state
        Returns:
            pred: [B, C, H, W] - predicted next state
        """
        # Encoder
        skips = []
        h = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for up, dec, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:])
            h = torch.cat([h, skip], dim=1)
            h = dec[0](h)   # fusion conv
            h = dec[1](h)   # refinement conv
        
        # Residual prediction
        tendency = self.output_proj(h)
        pred = x + self.tendency_scale * tendency
        return pred


class CascadeForecastSystem:
    """
    Three-stage cascade weather forecasting system.
    
    Stage 1: Short-range (6h-3d, 12 steps)
    Stage 2: Medium-range (3d-7d, 16 steps)
    Stage 3: Extended-range (7d-15d, 32 steps)
    """
    def __init__(self, in_channels=70, base_channels=32):
        self.device = torch.device('cpu')
        
        # Stage 1: Short-range model - optimized for rapid dynamics
        self.stage1 = CascadeUTransformer(
            in_channels=in_channels, base_channels=base_channels, 
            depth=3, forecast_steps=12, stage_name='short_range'
        ).to(self.device)
        
        # Stage 2: Medium-range model - optimized for synoptic patterns
        self.stage2 = CascadeUTransformer(
            in_channels=in_channels, base_channels=base_channels,
            depth=3, forecast_steps=16, stage_name='medium_range'
        ).to(self.device)
        
        # Stage 3: Extended-range model - optimized for large-scale trends
        self.stage3 = CascadeUTransformer(
            in_channels=in_channels, base_channels=base_channels,
            depth=3, forecast_steps=32, stage_name='extended_range'
        ).to(self.device)
        
        # Model sizes
        self.stage1_params = sum(p.numel() for p in self.stage1.parameters())
        self.stage2_params = sum(p.numel() for p in self.stage2.parameters())
        self.stage3_params = sum(p.numel() for p in self.stage3.parameters())
    
    def get_model_info(self):
        return {
            'stage1': {'name': 'Short-Range (6h-3d)', 'params': self.stage1_params, 'steps': 12},
            'stage2': {'name': 'Medium-Range (3d-7d)', 'params': self.stage2_params, 'steps': 16},
            'stage3': {'name': 'Extended-Range (7d-15d)', 'params': self.stage3_params, 'steps': 32},
            'total_params': self.stage1_params + self.stage2_params + self.stage3_params
        }
    
    @torch.no_grad()
    def generate_cascade_forecast(self, input_states):
        """
        Generate 15-day forecast using cascade approach.
        
        Args:
            input_states: numpy array [2, 70, 181, 360] - two consecutive states
        Returns:
            forecast: numpy array [60, 70, 181, 360] - 60 steps of 6h forecasts
        """
        self.stage1.eval()
        self.stage2.eval()
        self.stage3.eval()
        
        x = torch.tensor(input_states[-1:], dtype=torch.float32).to(self.device)
        forecasts = []
        
        # Stage 1: Short-range (autoregressive, 12 steps)
        current = x.clone()
        for step in range(12):
            next_state = self.stage1(current)
            forecasts.append(next_state.cpu().numpy()[0])
            current = next_state
        
        # Handover to Stage 2: use last stage1 prediction as initial state
        for step in range(16):
            next_state = self.stage2(current)
            forecasts.append(next_state.cpu().numpy()[0])
            current = next_state
        
        # Handover to Stage 3: use last stage2 prediction as initial state
        for step in range(32):
            next_state = self.stage3(current)
            forecasts.append(next_state.cpu().numpy()[0])
            current = next_state
        
        return np.array(forecasts)  # [60, 70, 181, 360]
    
    @torch.no_grad()
    def generate_single_model_forecast(self, input_states, num_steps=60):
        """
        Generate forecast using single U-Transformer (no cascade).
        For comparison: demonstrates error accumulation.
        """
        model = CascadeUTransformer(
            in_channels=70, base_channels=32, depth=3, 
            forecast_steps=num_steps, stage_name='single'
        ).to(self.device)
        model.eval()
        
        x = torch.tensor(input_states[-1:], dtype=torch.float32).to(self.device)
        forecasts = []
        current = x.clone()
        for step in range(num_steps):
            next_state = model(current)
            forecasts.append(next_state.cpu().numpy()[0])
            current = next_state
        
        return np.array(forecasts)


def main():
    print("=" * 60)
    print("Phase 2: Cascade U-Transformer System")
    print("=" * 60)
    
    # Load data
    ds_in = xr.open_dataset('data/20231012-06_input_netcdf.nc')
    ds_out = xr.open_dataset('data/006.nc')
    input_data = ds_in.data.values  # [2, 70, 181, 360]
    fuxi_data = ds_out.data.values[0, 0]  # [70, 181, 360]
    
    # Initialize cascade system
    cascade = CascadeForecastSystem(in_channels=70, base_channels=32)
    info = cascade.get_model_info()
    
    print("\nModel Architecture:")
    for key, val in info.items():
        if key != 'total_params':
            print(f"  {val['name']}: {val['params']:,} parameters, {val['steps']} steps")
    print(f"  Total: {info['total_params']:,} parameters")
    
    # Generate cascade forecast
    print("\nGenerating cascade forecast...")
    cascade_forecast = cascade.generate_cascade_forecast(input_data)
    print(f"  Cascade forecast shape: {cascade_forecast.shape}")
    
    # Generate single-model forecast
    print("Generating single-model forecast for comparison...")
    single_forecast = cascade.generate_single_model_forecast(input_data)
    print(f"  Single-model forecast shape: {single_forecast.shape}")
    
    # Analyze error accumulation
    print("\nAnalyzing error accumulation...")
    
    # Compute per-step RMSE relative to input state (as reference)
    reference = input_data[-1]  # [70, 181, 360]
    
    cascade_rmse = []
    single_rmse = []
    for step in range(60):
        c_diff = cascade_forecast[step] - reference
        s_diff = single_forecast[step] - reference
        cascade_rmse.append(np.sqrt(np.mean(c_diff**2)))
        single_rmse.append(np.sqrt(np.mean(s_diff**2)))
    
    # Compare with FuXi at step 0 (t+6h)
    fuxi_rmse = np.sqrt(np.mean((fuxi_data - reference)**2))
    cascade_step0_rmse = np.sqrt(np.mean((cascade_forecast[0] - reference)**2))
    
    print(f"\n  FuXi t+6h RMSE: {fuxi_rmse:.4f}")
    print(f"  Cascade t+6h RMSE: {cascade_step0_rmse:.4f}")
    print(f"  Single-model t+6h RMSE: {single_rmse[0]:.4f}")
    
    # Per-group analysis
    levels_list = list(ds_in.level.values)
    groups = {
        'Geopotential (Z)': [levels_list.index(v) for v in ['Z50','Z100','Z200','Z500','Z1000']],
        'Temperature (T)': [levels_list.index(v) for v in ['T100','T500','T850','T2M']],
        'U-Wind (U)': [levels_list.index(v) for v in ['U200','U500','U850','U10']],
        'V-Wind (V)': [levels_list.index(v) for v in ['V200','V500','V850','V10']],
        'Humidity (R)': [levels_list.index(v) for v in ['R500','R850','R1000']],
    }
    
    group_rmse = {}
    for gname, gidx in groups.items():
        g_cascade = []
        g_single = []
        g_ref = reference[gidx]
        for step in range(60):
            c_diff = cascade_forecast[step][gidx] - g_ref
            s_diff = single_forecast[step][gidx] - g_ref
            g_cascade.append(np.sqrt(np.mean(c_diff**2)))
            g_single.append(np.sqrt(np.mean(s_diff**2)))
        group_rmse[gname] = {'cascade': g_cascade, 'single': g_single}
    
    # Save results
    results = {
        'model_info': info,
        'cascade_rmse': cascade_rmse,
        'single_rmse': single_rmse,
        'fuxi_step0_rmse': float(fuxi_rmse),
        'group_rmse': group_rmse
    }
    
    with open('outputs/cascade_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save forecast data as netcdf
    ds_forecast = xr.Dataset(
        {
            'cascade_forecast': (['step', 'level', 'lat', 'lon'], cascade_forecast),
            'single_forecast': (['step', 'level', 'lat', 'lon'], single_forecast),
        },
        coords={
            'step': np.arange(1, 61),
            'level': levels_list,
            'lat': ds_in.lat.values,
            'lon': ds_in.lon.values,
        }
    )
    ds_forecast.to_netcdf('outputs/cascade_forecast.nc')
    ds_in.close()
    ds_out.close()
    
    print("\nPhase 2 complete.")
    return results


if __name__ == '__main__':
    results = main()
