"""
Lightweight U-Net model for CPU-based demonstration.
U-Transformer blocks replaced with efficient conv blocks for training feasibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.skip(x)


class LightUNet(nn.Module):
    """Lightweight U-Net for weather forecasting."""
    def __init__(self, in_channels=140, out_channels=70, base_ch=16):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, 70, 1) if in_channels != 70 else nn.Identity()
        
        # Encoder
        self.enc1 = ConvBlock(70, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        
        self.final = nn.Conv2d(base_ch, out_channels, 1)
        
    def forward(self, x):
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b, t * c, h, w)
        x = self.input_proj(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        d2 = self.up2(e3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.final(d1)


class CascadeUNet(nn.Module):
    """Cascade of three specialized U-Net models."""
    def __init__(self, in_channels=140, out_channels=70, base_ch=16):
        super().__init__()
        self.stage1 = LightUNet(in_channels, out_channels, base_ch)
        self.stage2 = LightUNet(in_channels, out_channels, base_ch)
        self.stage3 = LightUNet(in_channels, out_channels, base_ch)
        
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
    def __init__(self, in_channels=140, out_channels=70, base_ch=16):
        super().__init__()
        self.model = LightUNet(in_channels, out_channels, base_ch)
    
    def forward(self, x):
        return self.model(x)
    
    def rollout(self, x0, num_steps=60):
        forecasts = []
        x = x0
        for _ in range(num_steps):
            x = self.forward(x)
            forecasts.append(x.detach().cpu())
        return forecasts
