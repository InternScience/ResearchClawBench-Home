"""
Workflow #2 — Structure (lattice) generation with a Variational Autoencoder.

Inputs: 101 (a, b) lattice-parameter pairs (parsed from M-AI-Synth block 2).
Model: a small VAE with encoder q_phi(z|x) and decoder p_theta(x|z),
       z in R^2, MLPs of size 32, ELBO = recon (MSE) + beta*KL.
Outputs:
  * trained VAE
  * 1000 generated (a*, b*) samples
  * Wasserstein-1 and KS distances vs the training distribution
  * coverage (% of generated samples inside the training bounding box).
We also include a Gaussian-fit baseline (sampling from a 2-D normal
fit to the training data) to sanity-check the VAE.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
torch.manual_seed(1)
np.random.seed(1)


class VAE(nn.Module):
    def __init__(self, x_dim=2, z_dim=2, h=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z_dim)
        self.lv = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, x_dim))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.dec(z), mu, lv


def main():
    npz = np.load(OUT / "parsed_data.npz")
    a = npz["sg_a"].astype(np.float32)
    b = npz["sg_b"].astype(np.float32)
    X = np.stack([a, b], axis=1)  # (101, 2)

    # standardize
    mu_x, sigma_x = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu_x) / sigma_x
    Xt = torch.from_numpy(Xs)

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    beta = 1.0
    history = {"loss": [], "recon": [], "kl": []}
    n_epochs = 1500
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        recon, mu, lv = model(Xt)
        rec = F.mse_loss(recon, Xt, reduction="mean")
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss = rec + beta * kl
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.item()))
        history["recon"].append(float(rec.item()))
        history["kl"].append(float(kl.item()))

    # generate
    model.eval()
    with torch.no_grad():
        z = torch.randn(1000, 2)
        Xg = model.dec(z).numpy() * sigma_x + mu_x

    # Gaussian baseline
    rng = np.random.default_rng(0)
    cov = np.cov(X.T)
    Xg_gauss = rng.multivariate_normal(X.mean(0), cov, size=1000).astype(np.float32)

    def stats(samples):
        return {
            "ks_a": float(ks_2samp(a, samples[:, 0]).statistic),
            "ks_b": float(ks_2samp(b, samples[:, 1]).statistic),
            "w_a":  float(wasserstein_distance(a, samples[:, 0])),
            "w_b":  float(wasserstein_distance(b, samples[:, 1])),
            "in_bbox_pct": float(
                np.mean(
                    (samples[:, 0] >= a.min()) & (samples[:, 0] <= a.max()) &
                    (samples[:, 1] >= b.min()) & (samples[:, 1] <= b.max())
                ) * 100.0
            ),
            "mean_a": float(samples[:, 0].mean()),
            "mean_b": float(samples[:, 1].mean()),
            "std_a":  float(samples[:, 0].std()),
            "std_b":  float(samples[:, 1].std()),
        }

    metrics = {
        "n_real": int(len(a)),
        "real": {
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "std_a":  float(a.std()),  "std_b":  float(b.std()),
        },
        "VAE":      stats(Xg),
        "Gaussian": stats(Xg_gauss),
        "epochs": n_epochs,
        "final_loss": history["loss"][-1],
        "final_recon": history["recon"][-1],
        "final_kl": history["kl"][-1],
    }
    (OUT / "structure_generation_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(OUT / "structure_generation_samples.npz",
             real=X, vae=Xg, gauss=Xg_gauss,
             history_loss=np.array(history["loss"]),
             history_recon=np.array(history["recon"]),
             history_kl=np.array(history["kl"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
