"""
Train all cascade models efficiently.
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


class QuickDS(Dataset):
    def __init__(self, data_in, num_augments=30, noise_scale=0.05, down=4):
        self.x = data_in.astype(np.float32).reshape(-1, data_in.shape[-2], data_in.shape[-1])
        self.y = data_in[1].astype(np.float32)
        if down > 1:
            self.x = self.x[:, ::down, ::down]
            self.y = self.y[:, ::down, ::down]
        self.num_augments = num_augments
        self.noise_scale = noise_scale
        
    def __len__(self):
        return self.num_augments
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        nx = np.random.randn(*self.x.shape).astype(np.float32) * self.noise_scale
        ny = np.random.randn(*self.y.shape).astype(np.float32) * self.noise_scale
        return torch.from_numpy(self.x + nx), torch.from_numpy(self.y + ny)


def train_one(model, train_loader, val_loader, epochs=8, lr=3e-3, save_path='outputs/model.pt'):
    device = 'cpu'
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    best_val = float('inf')
    
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)
        
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vl += crit(model(x), y).item()
        vl /= len(val_loader)
        sched.step()
        
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)
        print(f"  E{epoch+1}: train={tl:.4f} val={vl:.4f}")
    
    return best_val


def main():
    data_in, _, _, _ = load_data()
    down = 4
    h, w = data_in.shape[-2] // down, data_in.shape[-1] // down
    print(f"Training at {h}x{w}")
    
    n = 30
    train_n = 25
    val_n = 5
    
    # Stage 1
    ds1 = QuickDS(data_in, num_augments=n, noise_scale=0.05, down=down)
    t1, v1 = torch.utils.data.random_split(ds1, [train_n, val_n])
    print("Stage 1...")
    m1 = LightUNet(140, 70, 16)
    b1 = train_one(m1, DataLoader(t1, 4, True), DataLoader(v1, 4), 8, 3e-3, 'outputs/stage1.pt')
    
    # Stage 2
    ds2 = QuickDS(data_in, num_augments=n, noise_scale=0.15, down=down)
    t2, v2 = torch.utils.data.random_split(ds2, [train_n, val_n])
    print("Stage 2...")
    m2 = LightUNet(140, 70, 16)
    b2 = train_one(m2, DataLoader(t2, 4, True), DataLoader(v2, 4), 8, 3e-3, 'outputs/stage2.pt')
    
    # Stage 3
    ds3 = QuickDS(data_in, num_augments=n, noise_scale=0.25, down=down)
    t3, v3 = torch.utils.data.random_split(ds3, [train_n, val_n])
    print("Stage 3...")
    m3 = LightUNet(140, 70, 16)
    b3 = train_one(m3, DataLoader(t3, 4, True), DataLoader(v3, 4), 8, 3e-3, 'outputs/stage3.pt')
    
    # Single
    print("Single...")
    ms = LightUNet(140, 70, 16)
    bs = train_one(ms, DataLoader(t1, 4, True), DataLoader(v1, 4), 8, 3e-3, 'outputs/single.pt')
    
    print(f"Done. Best vals: S1={b1:.4f} S2={b2:.4f} S3={b3:.4f} Single={bs:.4f}")


if __name__ == '__main__':
    main()
