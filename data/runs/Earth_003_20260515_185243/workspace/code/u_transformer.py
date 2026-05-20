import torch
import torch.nn as nn
import torch.nn.functional as F

class UTransformer(nn.Module):
    def __init__(self, in_channels=140, out_channels=70, hidden=64):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden, hidden*2, 3, padding=1, stride=2)
        self.bottleneck = nn.Conv2d(hidden*2, hidden*4, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(hidden*4, hidden*2, 2, stride=2)
        self.dec2 = nn.Conv2d(hidden*3, hidden, 3, padding=1)  # concat skip: 128+64=192
        self.out = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, x):
        # x: (B, 2*70, H, W)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        b = F.relu(self.bottleneck(e2))
        d1 = F.relu(self.dec1(b))
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.relu(self.dec2(torch.cat([d1, e1], dim=1)))  # concat skip
        return self.out(d2)