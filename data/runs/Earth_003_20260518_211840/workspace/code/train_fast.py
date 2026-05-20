"""
Fast training script with spatial downsampling for CPU feasibility.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data, compute_lat_weights
from model_light import LightUNet, CascadeUNet, SingleModelBaseline


class FastWeatherDataset(Dataset):
    """Dataset with 4x spatial downsampling for fast CPU training."""
    def __init__(self, data_in, num_augments=32, noise_scale=0.05, down_factor=4):
        self.data_in = data_in
        self.num_augments = num_augments
        self.noise_scale = noise_scale
        self.down_factor = down_factor
        
    def __len__(self):
        return self.num_augments
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        x = self.data_in.copy().astype(np.float32)
        y = self.data_in[1].copy().astype(np.float32)
        
        noise_x = np.random.randn(*x.shape).astype(np.float32) * self.noise_scale
        noise_y = np.random.randn(*y.shape).astype(np.float32) * self.noise_scale
        x = x + noise_x
        y = y + noise_y
        
        shift = np.random.randint(0, x.shape[-1])
        x = np.roll(x, shift, axis=-1)
        y = np.roll(y, shift, axis=-1)
        
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        
        # Flatten timesteps
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        # Downsample
        if self.down_factor > 1:
            x = x[:, ::self.down_factor, ::self.down_factor]
            y = y[:, ::self.down_factor, ::self.down_factor]
        
        return torch.from_numpy(x), torch.from_numpy(y)


class FastCascadeDataset(Dataset):
    def __init__(self, data_in, num_steps=5, num_augments=32, noise_scale=0.05, 
                 error_scale=0.1, down_factor=4):
        self.data_in = data_in
        self.num_steps = num_steps
        self.num_augments = num_augments
        self.noise_scale = noise_scale
        self.error_scale = error_scale
        self.down_factor = down_factor
        
    def __len__(self):
        return self.num_augments
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        x0 = self.data_in[0].copy().astype(np.float32)
        x0 = x0 + np.random.randn(*x0.shape).astype(np.float32) * self.noise_scale
        
        error = np.zeros_like(x0)
        for _ in range(self.num_steps):
            error += np.random.randn(*x0.shape).astype(np.float32) * self.error_scale
            from scipy.ndimage import gaussian_filter
            error = gaussian_filter(error, sigma=1.0)
        
        x = x0 + error
        y = self.data_in[1].copy().astype(np.float32) + error * 0.5
        
        x_full = np.stack([x, y], axis=0).reshape(-1, x.shape[-2], x.shape[-1])
        
        if self.down_factor > 1:
            x_full = x_full[:, ::self.down_factor, ::self.down_factor]
            y = y[:, ::self.down_factor, ::self.down_factor]
        
        return torch.from_numpy(x_full), torch.from_numpy(y)


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu',
                save_path='outputs/model.pt', early_stop_patience=3):
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
        
        print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, train_losses, val_losses


def main():
    device = 'cpu'
    down_factor = 4
    print(f"Using device: {device}, down_factor={down_factor}")
    
    data_in, data_fuxi, lats, lons = load_data()
    h, w = data_in.shape[-2] // down_factor, data_in.shape[-1] // down_factor
    print(f"Training resolution: {h}x{w}")
    
    ds1 = FastWeatherDataset(data_in, num_augments=48, noise_scale=0.05, down_factor=down_factor)
    train_size = int(0.85 * len(ds1))
    val_size = len(ds1) - train_size
    train_ds1, val_ds1 = torch.utils.data.random_split(ds1, [train_size, val_size])
    train_loader1 = DataLoader(train_ds1, batch_size=4, shuffle=True)
    val_loader1 = DataLoader(val_ds1, batch_size=4)
    
    ds2 = FastCascadeDataset(data_in, num_steps=20, num_augments=48, noise_scale=0.05,
                             error_scale=0.08, down_factor=down_factor)
    train_ds2, val_ds2 = torch.utils.data.random_split(ds2, [train_size, val_size])
    train_loader2 = DataLoader(train_ds2, batch_size=4, shuffle=True)
    val_loader2 = DataLoader(val_ds2, batch_size=4)
    
    ds3 = FastCascadeDataset(data_in, num_steps=40, num_augments=48, noise_scale=0.05,
                             error_scale=0.15, down_factor=down_factor)
    train_ds3, val_ds3 = torch.utils.data.random_split(ds3, [train_size, val_size])
    train_loader3 = DataLoader(train_ds3, batch_size=4, shuffle=True)
    val_loader3 = DataLoader(val_ds3, batch_size=4)
    
    print("\n=== Stage 1 ===")
    stage1 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    stage1, tl1, vl1 = train_model(stage1, train_loader1, val_loader1, epochs=12,
                                    lr=3e-3, device=device, save_path='outputs/stage1.pt')
    
    print("\n=== Stage 2 ===")
    stage2 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    stage2, tl2, vl2 = train_model(stage2, train_loader2, val_loader2, epochs=12,
                                    lr=3e-3, device=device, save_path='outputs/stage2.pt')
    
    print("\n=== Stage 3 ===")
    stage3 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    stage3, tl3, vl3 = train_model(stage3, train_loader3, val_loader3, epochs=12,
                                    lr=3e-3, device=device, save_path='outputs/stage3.pt')
    
    print("\n=== Single Baseline ===")
    single = SingleModelBaseline(in_channels=140, out_channels=70, base_ch=16)
    single, tls, vls = train_model(single, train_loader1, val_loader1, epochs=12,
                                    lr=3e-3, device=device, save_path='outputs/single.pt')
    
    np.savez('outputs/training_curves.npz',
             stage1_train=tl1, stage1_val=vl1,
             stage2_train=tl2, stage2_val=vl2,
             stage3_train=tl3, stage3_val=vl3,
             single_train=tls, single_val=vls)
    print("\nDone!")
    print(f"S1: {min(vl1):.4f}, S2: {min(vl2):.4f}, S3: {min(vl3):.4f}, Single: {min(vls):.4f}")


if __name__ == '__main__':
    main()
