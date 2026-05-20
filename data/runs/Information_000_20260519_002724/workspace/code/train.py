"""
Training script for VQ-VAE and Unified Transformer.
"""
import os
import json
import math
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vqvae import VQVAE
from unified_model import UnifiedTransformer, UnderstandingEncoder, CoupledEncoder
from tokenizer import encode, encode_raw, decode, PAD_ID, SOS_ID, EOS_ID, IMG_START_ID, IMG_END_ID, IMG_TOKEN_START, TOTAL_VOCAB


# ------------------- Config -------------------
DEVICE = torch.device('cpu')
BATCH_SIZE = 64
LR = 1e-3
VQVAE_EPOCHS = 25
UNIFIED_EPOCHS = 40
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
D_FF = 512
MAX_GEN_LEN = 100
MAX_TXT_LEN = 50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------- Dataset -------------------
class SyntheticDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        with open(os.path.join(root_dir, 'meta.json'), 'r') as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img_path = os.path.join(self.root, f"img_{idx:05d}.png")
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img, item


def collate_vqvae(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs


def collate_generation(batch, vqvae, max_len=MAX_GEN_LEN):
    # batch: list of (img, meta)
    imgs = torch.stack([b[0] for b in batch])
    with torch.no_grad():
        _, _, indices = vqvae.encode(imgs)
    B = len(batch)
    seqs = []
    for i, (img, meta) in enumerate(batch):
        text_ids = encode_raw(meta['caption'], max_len=18)
        img_ids = indices[i].flatten().tolist()
        seq = [SOS_ID] + text_ids + [IMG_START_ID] + img_ids + [IMG_END_ID, EOS_ID]
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq = seq + [PAD_ID] * (max_len - len(seq))
        seqs.append(seq)
    seqs = torch.tensor(seqs, dtype=torch.long)
    return seqs


def collate_understanding(batch, max_len=MAX_TXT_LEN):
    imgs = torch.stack([b[0] for b in batch])
    # randomly pick a QA pair
    seqs = []
    for img, meta in batch:
        q, a = random.choice(meta['qa'])
        q_ids = encode_raw(q, max_len=20)
        a_ids = encode_raw(a, max_len=15)
        seq = [SOS_ID] + q_ids + a_ids + [EOS_ID]
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq = seq + [PAD_ID] * (max_len - len(seq))
        seqs.append(seq)
    seqs = torch.tensor(seqs, dtype=torch.long)
    return imgs, seqs


# ------------------- Training VQ-VAE -------------------
def train_vqvae(train_loader, val_loader):
    model = VQVAE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val = float('inf')
    history = {'train': [], 'val': []}
    for epoch in range(1, VQVAE_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs in tqdm(train_loader, desc=f"VQVAE Epoch {epoch}"):
            imgs = imgs.to(DEVICE)
            _, loss, rloss, vqloss, _ = model(imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(DEVICE)
                _, loss, _, _, _ = model(imgs)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'outputs/vqvae_best.pt')
    return model, history


# ------------------- Training Unified Model -------------------
def train_unified(train_loader, val_loader, vqvae, use_decoupled=True):
    transformer = UnifiedTransformer(vocab_size=TOTAL_VOCAB, d_model=D_MODEL,
                                     n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF).to(DEVICE)
    if use_decoupled:
        vis_enc = UnderstandingEncoder(d_model=D_MODEL).to(DEVICE)
    else:
        vis_enc = CoupledEncoder(vqvae.encoder, vqvae.quantizer, d_model=D_MODEL).to(DEVICE)

    params = list(transformer.parameters()) + list(vis_enc.parameters())
    opt = torch.optim.AdamW(params, lr=LR)

    best_val = float('inf')
    history = {'gen_train': [], 'gen_val': [], 'under_train': [], 'under_val': []}

    for epoch in range(1, UNIFIED_EPOCHS + 1):
        transformer.train()
        vis_enc.train()
        gen_loss_sum = 0.0
        under_loss_sum = 0.0
        gen_steps = 0
        under_steps = 0

        for batch in tqdm(train_loader, desc=f"Unified Epoch {epoch}"):
            # alternate tasks
            if random.random() < 0.5:
                # Generation
                seqs = collate_generation(batch, vqvae).to(DEVICE)
                inp = seqs[:, :-1]
                tgt = seqs[:, 1:]
                logits = transformer(inp)
                logits = logits.reshape(-1, logits.size(-1))
                tgt_flat = tgt.reshape(-1)
                loss = F.cross_entropy(logits, tgt_flat, ignore_index=PAD_ID)
                opt.zero_grad()
                loss.backward()
                opt.step()
                gen_loss_sum += loss.item()
                gen_steps += 1
            else:
                # Understanding
                imgs, seqs = collate_understanding(batch)
                imgs = imgs.to(DEVICE)
                seqs = seqs.to(DEVICE)
                inp = seqs[:, :-1]
                tgt = seqs[:, 1:]
                vis = vis_enc(imgs)
                logits = transformer(inp, continuous_prefix=vis)
                logits = logits[:, vis.size(1):, :]
                logits = logits.reshape(-1, logits.size(-1))
                tgt_flat = tgt.reshape(-1)
                loss = F.cross_entropy(logits, tgt_flat, ignore_index=PAD_ID)
                opt.zero_grad()
                loss.backward()
                opt.step()
                under_loss_sum += loss.item()
                under_steps += 1

        history['gen_train'].append(gen_loss_sum / max(gen_steps, 1))
        history['under_train'].append(under_loss_sum / max(under_steps, 1))

        # Validation
        transformer.eval()
        vis_enc.eval()
        gen_loss_sum = 0.0
        under_loss_sum = 0.0
        gen_steps = 0
        under_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                # gen
                seqs = collate_generation(batch, vqvae).to(DEVICE)
                inp = seqs[:, :-1]
                tgt = seqs[:, 1:]
                logits = transformer(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD_ID)
                gen_loss_sum += loss.item()
                gen_steps += 1
                # under
                imgs, seqs = collate_understanding(batch)
                imgs = imgs.to(DEVICE)
                seqs = seqs.to(DEVICE)
                inp = seqs[:, :-1]
                vis = vis_enc(imgs)
                logits = transformer(inp, continuous_prefix=vis)
                logits = logits[:, vis.size(1):, :]
                tgt = seqs[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD_ID)
                under_loss_sum += loss.item()
                under_steps += 1
        history['gen_val'].append(gen_loss_sum / max(gen_steps, 1))
        history['under_val'].append(under_loss_sum / max(under_steps, 1))
        val_total = history['gen_val'][-1] + history['under_val'][-1]
        print(f"Epoch {epoch}: gen_train={history['gen_train'][-1]:.4f} under_train={history['under_train'][-1]:.4f} "
              f"gen_val={history['gen_val'][-1]:.4f} under_val={history['under_val'][-1]:.4f}")
        if val_total < best_val:
            best_val = val_total
            torch.save({
                'transformer': transformer.state_dict(),
                'vis_enc': vis_enc.state_dict(),
                'use_decoupled': use_decoupled,
            }, 'outputs/unified_decoupled.pt' if use_decoupled else 'outputs/unified_coupled.pt')
    return transformer, vis_enc, history


# ------------------- Main -------------------
if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    train_ds = SyntheticDataset('outputs/synthetic_train')
    val_ds = SyntheticDataset('outputs/synthetic_val')
    test_ds = SyntheticDataset('outputs/synthetic_test')
    train_loader_vqvae = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_vqvae)
    val_loader_vqvae = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_vqvae)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

    # Train VQ-VAE
    print("Training VQ-VAE...")
    vqvae, vqvae_hist = train_vqvae(train_loader_vqvae, val_loader_vqvae)
    vqvae.load_state_dict(torch.load('outputs/vqvae_best.pt', map_location=DEVICE))
    vqvae.eval()

    # Save VQVAE history
    with open('outputs/vqvae_history.json', 'w') as f:
        json.dump(vqvae_hist, f)

    # Train Decoupled Unified Model
    print("Training Decoupled Unified Model...")
    transformer_dec, vis_enc_dec, hist_dec = train_unified(train_loader, val_loader, vqvae, use_decoupled=True)
    with open('outputs/unified_decoupled_history.json', 'w') as f:
        json.dump(hist_dec, f)

    # Train Coupled Unified Model
    print("Training Coupled Unified Model...")
    transformer_coup, vis_enc_coup, hist_coup = train_unified(train_loader, val_loader, vqvae, use_decoupled=False)
    with open('outputs/unified_coupled_history.json', 'w') as f:
        json.dump(hist_coup, f)

    print("Training complete.")
