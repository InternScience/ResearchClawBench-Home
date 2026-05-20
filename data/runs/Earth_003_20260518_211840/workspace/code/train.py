"""
Training script for cascade U-Transformer weather forecasting.
Uses lightweight U-Net for CPU feasibility.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data, WeatherDataset, CascadeWeatherDataset
from model_light import LightUNet, CascadeUNet, SingleModelBaseline


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3, device='cpu',
                save_path='outputs/model.pt', early_stop_patience=4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, train_losses, val_losses


def main():
    device = 'cpu'
    print(f"Using device: {device}")
    
    data_in, data_fuxi, lats, lons = load_data()
    print(f"Input shape: {data_in.shape}")
    
    ds1 = WeatherDataset(data_in, num_augments=80, noise_scale=0.05)
    train_size = int(0.85 * len(ds1))
    val_size = len(ds1) - train_size
    train_ds1, val_ds1 = torch.utils.data.random_split(ds1, [train_size, val_size])
    train_loader1 = DataLoader(train_ds1, batch_size=4, shuffle=True, num_workers=0)
    val_loader1 = DataLoader(val_ds1, batch_size=4, num_workers=0)
    
    ds2 = CascadeWeatherDataset(data_in, num_steps=20, num_augments=80,
                                 noise_scale=0.05, error_scale=0.08)
    train_ds2, val_ds2 = torch.utils.data.random_split(ds2, [train_size, val_size])
    train_loader2 = DataLoader(train_ds2, batch_size=4, shuffle=True, num_workers=0)
    val_loader2 = DataLoader(val_ds2, batch_size=4, num_workers=0)
    
    ds3 = CascadeWeatherDataset(data_in, num_steps=40, num_augments=80,
                                 noise_scale=0.05, error_scale=0.15)
    train_ds3, val_ds3 = torch.utils.data.random_split(ds3, [train_size, val_size])
    train_loader3 = DataLoader(train_ds3, batch_size=4, shuffle=True, num_workers=0)
    val_loader3 = DataLoader(val_ds3, batch_size=4, num_workers=0)
    
    print("\n=== Training Stage 1 (Short-range) ===")
    stage1 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    stage1, tl1, vl1 = train_model(stage1, train_loader1, val_loader1, epochs=15,
                                    lr=3e-3, device=device,
                                    save_path='outputs/stage1.pt')
    
    print("\n=== Training Stage 2 (Medium-range) ===")
    stage2 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    stage2, tl2, vl2 = train_model(stage2, train_loader2, val_loader2, epochs=15,
                                    lr=3e-3, device=device,
                                    save_path='outputs/stage2.pt')
    
    print("\n=== Training Stage 3 (Long-range) ===")
    stage3 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    stage3, tl3, vl3 = train_model(stage3, train_loader3, val_loader3, epochs=15,
                                    lr=3e-3, device=device,
                                    save_path='outputs/stage3.pt')
    
    print("\n=== Training Single Model Baseline ===")
    single = SingleModelBaseline(in_channels=140, out_channels=70, base_ch=16)
    single, tls, vls = train_model(single, train_loader1, val_loader1, epochs=15,
                                    lr=3e-3, device=device,
                                    save_path='outputs/single.pt')
    
    np.savez('outputs/training_curves.npz',
             stage1_train=tl1, stage1_val=vl1,
             stage2_train=tl2, stage2_val=vl2,
             stage3_train=tl3, stage3_val=vl3,
             single_train=tls, single_val=vls)
    
    print("\nTraining complete!")
    print(f"Stage 1 best: {min(vl1):.6f}")
    print(f"Stage 2 best: {min(vl2):.6f}")
    print(f"Stage 3 best: {min(vl3):.6f}")
    print(f"Single best: {min(vls):.6f}")


if __name__ == '__main__':
    main()
