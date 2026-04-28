"""Self-supervised pre-training: masked node feature recovery + graph-level
NT-Xent contrastive loss across two stochastic feature-mask views.
"""
import os, sys, json, copy, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models import GNNEncoder, PretrainHead, load_dataset, ROOT

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def make_view(batch, mask_prob: float = 0.2):
    b = batch.clone()
    # cache original one-hot for recon target on masked nodes
    orig = b.x.clone()
    n = b.x.size(0)
    mask = torch.rand(n, device=b.x.device) < mask_prob
    if mask.any():
        b.x = b.x.clone()
        b.x[mask] = 0.0  # erase atom identity
    return b, orig, mask


def nt_xent(z1, z2, tau: float = 0.2):
    # InfoNCE on a 2N batch
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / tau
    n = z1.size(0)
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, -1e9)
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def main():
    ds = load_dataset("pretrain_data.pt")
    loader = DataLoader(list(ds), batch_size=128, shuffle=True)

    enc = GNNEncoder().to(device)
    head = PretrainHead(enc.hidden).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-5)

    EPOCHS = 25
    history = []
    for ep in range(EPOCHS):
        enc.train(); head.train()
        tot, n_seen, recon_loss_sum, contr_loss_sum = 0.0, 0, 0.0, 0.0
        for batch in loader:
            batch = batch.to(device)
            v1, orig1, m1 = make_view(batch, 0.2)
            v2, orig2, m2 = make_view(batch, 0.2)
            g1, h1 = enc(v1.x, v1.edge_index, v1.edge_attr, v1.batch)
            g2, h2 = enc(v2.x, v2.edge_index, v2.edge_attr, v2.batch)

            # masked-feature reconstruction (BCE on one-hot)
            r1 = head.recon(h1[m1]); t1 = orig1[m1]
            r2 = head.recon(h2[m2]); t2 = orig2[m2]
            l_recon = 0.0
            if r1.numel() > 0:
                l_recon = l_recon + F.binary_cross_entropy_with_logits(r1, t1)
            if r2.numel() > 0:
                l_recon = l_recon + F.binary_cross_entropy_with_logits(r2, t2)

            z1 = head.proj(g1); z2 = head.proj(g2)
            l_contr = nt_xent(z1, z2)

            loss = l_contr + 0.5 * l_recon
            opt.zero_grad(); loss.backward(); opt.step()
            bs = batch.num_graphs
            tot += float(loss.item()) * bs; n_seen += bs
            recon_loss_sum += float(l_recon if isinstance(l_recon, float) else l_recon.item()) * bs
            contr_loss_sum += float(l_contr.item()) * bs

        avg = tot / n_seen
        avg_r = recon_loss_sum / n_seen; avg_c = contr_loss_sum / n_seen
        history.append({"epoch": ep, "loss": avg, "recon": avg_r, "contrastive": avg_c})
        print(f"[pretrain] epoch={ep:02d}  loss={avg:.4f}  recon={avg_r:.4f}  contr={avg_c:.4f}")

    torch.save({"encoder": enc.state_dict(), "hidden": enc.hidden},
               os.path.join(OUT, "pretrained_encoder.pt"))
    with open(os.path.join(OUT, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved pretrained encoder.")


if __name__ == "__main__":
    main()
