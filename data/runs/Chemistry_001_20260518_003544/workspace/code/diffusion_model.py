"""
Minimal diffusion model for protein-ligand complex structure prediction.
Uses a simple score-based diffusion for ligand pose.
"""
import torch
import torch.nn as nn
import numpy as np
from data_loader import load_protein_ca, load_ligand_sdf

class SimpleDiffusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embed = nn.Embedding(1000, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.ctx_proj = nn.Linear(3, hidden_dim)

    def forward(self, x, t, protein_context):
        x_emb = self.encoder(x)
        t_emb = self.time_embed(t)
        ctx_emb = self.ctx_proj(protein_context)
        ctx = ctx_emb.mean(dim=0, keepdim=True).expand(x_emb.shape[0], -1)
        t_emb = t_emb.expand(x_emb.shape[0], -1) if t_emb.dim() == 1 else t_emb
        combined = torch.cat([x_emb, ctx], dim=-1)
        return self.decoder(combined)

def diffusion_loss(model, x0, protein_ca, num_steps=100):
    """Simple diffusion training loss (noise prediction)."""
    batch_size = x0.shape[0]
    t = torch.randint(0, num_steps, (batch_size,))
    noise = torch.randn_like(x0)
    alpha = torch.linspace(0.999, 0.01, num_steps)
    noisy_x = torch.sqrt(alpha[t])[:, None] * x0 + torch.sqrt(1 - alpha[t])[:, None] * noise
    pred_noise = model(noisy_x, t, protein_ca)
    return nn.MSELoss()(pred_noise, noise)

def sample_pose(model, protein_ca, num_steps=100):
    """Generate ligand pose using reverse diffusion."""
    x = torch.randn(21, 3)  # assume 21 atoms for ligand
    for t in reversed(range(num_steps)):
        t_tensor = torch.tensor([t])
        pred = model(x, t_tensor, protein_ca)
        x = x - pred * 0.01  # simple step
    return x.detach().numpy()

if __name__ == "__main__":
    pdb_path = "data/sample/2l3r/2l3r_protein.pdb"
    sdf_path = "data/sample/2l3r/2l3r_ligand.sdf"
    ca = torch.tensor(load_protein_ca(pdb_path), dtype=torch.float32)
    _, lig = load_ligand_sdf(sdf_path)
    lig = torch.tensor(lig[:21], dtype=torch.float32)  # truncate to 21 atoms
    model = SimpleDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        loss = diffusion_loss(model, lig, ca)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    pred_pose = sample_pose(model, ca)
    print(f"Predicted pose shape: {pred_pose.shape}")
    np.save("outputs/predicted_ligand_pose.npy", pred_pose)
