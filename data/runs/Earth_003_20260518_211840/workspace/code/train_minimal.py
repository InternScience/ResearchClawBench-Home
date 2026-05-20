"""
Minimal fast training - trains one lightweight model for demonstration.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data
from model_light import LightUNet


class SimpleDataset(Dataset):
    def __init__(self, data_in, num_augments=40, noise_scale=0.05, down_factor=4):
        self.data_in = data_in.astype(np.float32)
        self.num_augments = num_augments
        self.noise_scale = noise_scale
        self.down_factor = down_factor
        
    def __len__(self):
        return self.num_augments
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        x = self.data_in.copy()
        y = self.data_in[1].copy()
        
        x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_scale
        y = y + np.random.randn(*y.shape).astype(np.float32) * self.noise_scale
        
        shift = np.random.randint(0, x.shape[-1])
        x = np.roll(x, shift, axis=-1)
        y = np.roll(y, shift, axis=-1)
        
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        if self.down_factor > 1:
            x = x[:, ::self.down_factor, ::self.down_factor]
            y = y[:, ::self.down_factor, ::self.down_factor]
        
        return torch.from_numpy(x), torch.from_numpy(y)


def main():
    device = 'cpu'
    down_factor = 4
    data_in, _, _, _ = load_data()
    
    ds = SimpleDataset(data_in, num_augments=40, noise_scale=0.05, down_factor=down_factor)
    train_size = 34
    val_size = 6
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    
    h, w = data_in.shape[-2] // down_factor, data_in.shape[-1] // down_factor
    print(f"Resolution: {h}x{w}")
    
    # Train stage 1
    print("Training Stage 1...")
    model1 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    model1 = model1.to(device)
    opt = torch.optim.Adam(model1.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    crit = nn.MSELoss()
    
    best_val = float('inf')
    for epoch in range(10):
        model1.train()
        tl = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model1(x)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)
        
        model1.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model1(x)
                vl += crit(pred, y).item()
        vl /= len(val_loader)
        sched.step()
        
        if vl < best_val:
            best_val = vl
            torch.save(model1.state_dict(), 'outputs/stage1.pt')
        
        print(f"  Epoch {epoch+1}: train={tl:.4f}, val={vl:.4f}")
    
    # Train stage 2 (with more noise)
    print("Training Stage 2...")
    ds2 = SimpleDataset(data_in, num_augments=40, noise_scale=0.15, down_factor=down_factor)
    train_ds2, val_ds2 = torch.utils.data.random_split(ds2, [train_size, val_size])
    train_loader2 = DataLoader(train_ds2, batch_size=4, shuffle=True)
    val_loader2 = DataLoader(val_ds2, batch_size=4)
    
    model2 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    model2 = model2.to(device)
    opt = torch.optim.Adam(model2.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    
    best_val = float('inf')
    for epoch in range(10):
        model2.train()
        tl = 0.0
        for x, y in train_loader2:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model2(x)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader2)
        
        model2.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader2:
                x, y = x.to(device), y.to(device)
                pred = model2(x)
                vl += crit(pred, y).item()
        vl /= len(val_loader2)
        sched.step()
        
        if vl < best_val:
            best_val = vl
            torch.save(model2.state_dict(), 'outputs/stage2.pt')
        
        print(f"  Epoch {epoch+1}: train={tl:.4f}, val={vl:.4f}")
    
    # Train stage 3 (with even more noise)
    print("Training Stage 3...")
    ds3 = SimpleDataset(data_in, num_augments=40, noise_scale=0.25, down_factor=down_factor)
    train_ds3, val_ds3 = torch.utils.data.random_split(ds3, [train_size, val_size])
    train_loader3 = DataLoader(train_ds3, batch_size=4, shuffle=True)
    val_loader3 = DataLoader(val_ds3, batch_size=4)
    
    model3 = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    model3 = model3.to(device)
    opt = torch.optim.Adam(model3.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    
    best_val = float('inf')
    for epoch in range(10):
        model3.train()
        tl = 0.0
        for x, y in train_loader3:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model3(x)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model3.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader3)
        
        model3.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader3:
                x, y = x.to(device), y.to(device)
                pred = model3(x)
                vl += crit(pred, y).item()
        vl /= len(val_loader3)
        sched.step()
        
        if vl < best_val:
            best_val = vl
            torch.save(model3.state_dict(), 'outputs/stage3.pt')
        
        print(f"  Epoch {epoch+1}: train={tl:.4f}, val={vl:.4f}")
    
    # Train single baseline
    print("Training Single Baseline...")
    model_s = LightUNet(in_channels=140, out_channels=70, base_ch=16)
    model_s = model_s.to(device)
    opt = torch.optim.Adam(model_s.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    
    best_val = float('inf')
    for epoch in range(10):
        model_s.train()
        tl = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model_s(x)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)
        
        model_s.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model_s(x)
                vl += crit(pred, y).item()
        vl /= len(val_loader)
        sched.step()
        
        if vl < best_val:
            best_val = vl
            torch.save(model_s.state_dict(), 'outputs/single.pt')
        
        print(f"  Epoch {epoch+1}: train={tl:.4f}, val={vl:.4f}")
    
    print("All models trained!")


if __name__ == '__main__':
    main()
