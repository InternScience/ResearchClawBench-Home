"""
Demonstration of the Cascade U-Transformer training and inference pipeline.
"""
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'code')
from model import UTransformer, CascadeUTransformer


def demonstrate_model():
    """Demonstrate the U-Transformer model with small resolution."""
    print("=" * 60)
    print("Cascade U-Transformer Demonstration")
    print("=" * 60)
    
    # Use reduced resolution for CPU demonstration
    H, W = 45, 90  # 4x downsampled from 181x360
    in_channels = 140  # Two time steps concatenated (70+70)
    out_channels = 70
    base_dim = 32
    
    # Create single U-Transformer
    print("\n1. Single U-Transformer Model:")
    model = UTransformer(
        in_channels=in_channels, out_channels=out_channels,
        base_dim=base_dim, num_transformer_blocks=2,
        num_heads=4, input_h=H, input_w=W
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Forward pass
    x = torch.randn(1, in_channels, H, W)
    with torch.no_grad():
        out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Create Cascade U-Transformer
    print("\n2. Cascade U-Transformer System:")
    cascade = CascadeUTransformer(
        in_channels=in_channels, out_channels=out_channels,
        base_dim=base_dim, num_transformer_blocks=2,
        num_heads=4, input_h=H, input_w=W
    )
    n_params = sum(p.numel() for p in cascade.parameters())
    print(f"   Total parameters: {n_params:,}")
    
    # Stage-by-stage parameters
    for name, module in [('Stage 1', cascade.stage1), 
                          ('Stage 2', cascade.stage2), 
                          ('Stage 3', cascade.stage3)]:
        n = sum(p.numel() for p in module.parameters())
        print(f"   {name} parameters: {n:,}")
    
    # Single step inference for each stage
    x2 = torch.randn(1, in_channels, H, W)  # Two time steps concatenated
    with torch.no_grad():
        out1 = cascade.stage1(x2)
        out2 = cascade.stage2(x2)
        out3 = cascade.stage3(x2)
    print(f"\n   Stage 1 output: {out1.shape}")
    print(f"   Stage 2 output: {out2.shape}")
    print(f"   Stage 3 output: {out3.shape}")
    
    # Full autoregressive inference (just a few steps for demo)
    print("\n3. Autoregressive Inference (5 steps):")
    with torch.no_grad():
        forecasts = cascade(x2, max_steps=5)
    print(f"   Number of forecast steps: {len(forecasts)}")
    for i, f in enumerate(forecasts):
        print(f"   Step {i+1}: shape={f.shape}, mean={f.mean().item():.4f}")
    
    # Training demonstration
    print("\n4. Training Loop Demonstration:")
    optimizer = torch.optim.AdamW(cascade.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Synthetic training step
    x_input = torch.randn(2, in_channels, H, W)
    y_target = torch.randn(2, out_channels, H, W)
    
    for step in range(5):
        optimizer.zero_grad()
        output = cascade.stage1(x_input)
        loss = criterion(output, y_target)
        loss.backward()
        optimizer.step()
        print(f"   Step {step+1}: Loss = {loss.item():.4f}")
    
    print("\n5. Cascade Inference Pipeline:")
    print("   Stage 1: Steps 1-20 (0-5 days)")
    print("   Transition Layer 1: Error correction at day 5 boundary")
    print("   Stage 2: Steps 21-40 (5-10 days)")
    print("   Transition Layer 2: Error correction at day 10 boundary")
    print("   Stage 3: Steps 41-60 (10-15 days)")
    
    # Save model info
    model_info = {
        'total_params': n_params,
        'stage1_params': sum(p.numel() for p in cascade.stage1.parameters()),
        'stage2_params': sum(p.numel() for p in cascade.stage2.parameters()),
        'stage3_params': sum(p.numel() for p in cascade.stage3.parameters()),
        'transition1_params': sum(p.numel() for p in cascade.transition1.parameters()),
        'transition2_params': sum(p.numel() for p in cascade.transition2.parameters()),
        'input_resolution': f'{H}x{W}',
        'full_resolution': '181x360',
        'in_channels': in_channels,
        'out_channels': out_channels,
        'base_dim': base_dim,
    }
    
    import json
    with open('outputs/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel info saved to outputs/model_info.json")
    return model_info


if __name__ == "__main__":
    demonstrate_model()
