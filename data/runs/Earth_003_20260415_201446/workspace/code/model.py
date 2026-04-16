import torch
import torch.nn as nn

class UTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A simple dummy model to represent the U-Transformer architecture
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x is (B, C, H, W)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

class CascadeForecastingSystem(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 specialized models
        self.model_1 = UTransformer(70 * 2, 70) # Takes 2 steps of 70 vars, predicts 1 step
        self.model_2 = UTransformer(70 * 2, 70)
        self.model_3 = UTransformer(70 * 2, 70)
        
    def forward(self, x, steps):
        # x is (B, 2, 70, H, W)
        # Flatten time and channels for input
        B, T, C, H, W = x.shape
        outputs = []
        current_input = x.view(B, T*C, H, W)
        
        for i in range(steps):
            if i < 5: # First 5 steps (e.g., up to day 1.25)
                model = self.model_1
            elif i < 20: # Steps 5 to 20 (up to day 5)
                model = self.model_2
            else: # Steps 20+ (up to day 15)
                model = self.model_3
                
            out = model(current_input) # (B, 70, H, W)
            outputs.append(out.unsqueeze(1)) # (B, 1, 70, H, W)
            
            # Update input for next step
            # current_input has shape (B, 140, H, W)
            # We want to drop the oldest step and append the new output
            old_step_2 = current_input[:, 70:, :, :]
            current_input = torch.cat([old_step_2, out], dim=1)
            
        return torch.cat(outputs, dim=1)
