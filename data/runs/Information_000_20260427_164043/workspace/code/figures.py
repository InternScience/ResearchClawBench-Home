"""Generate all report figures."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mp

from data_utils import build_synthetic, load_real_images
from models import VQTokenizer, UnderstandingEncoder
from train import stack_images, CKPT_DIR

WORKSPACE = Path(__file__).resolve().parent.parent
IMG_DIR = WORKSPACE / "report" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT = WORKSPACE / "outputs"

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


# -------- Architecture diagram --------------------------------------------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    def box(x, y, w, h, txt, color):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                       fc=color, ec='black', lw=1.0))
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=9)

    def arrow(x1, y1, x2, y2, color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color=color))

    # Inputs
    box(0.2, 4.5, 1.4, 0.7, "Image\nx", "#fff7c8")
    box(0.2, 1.0, 1.4, 0.7, "Text\nprompt", "#fff7c8")

    # Two visual front-ends (decoupling)
    box(2.2, 5.1, 2.4, 0.8, "Understanding\nencoder (ViT/SigLIP)", "#cfe9ff")
    box(2.2, 4.0, 2.4, 0.8, "VQ tokenizer\n(LlamaGen-style)", "#ffd9d9")

    # Adapters
    box(5.2, 5.1, 1.4, 0.8, "Linear\nprojector", "#cfe9ff")
    box(5.2, 4.0, 1.4, 0.8, "Codebook\n→ token IDs", "#ffd9d9")

    # Text embed
    box(2.2, 1.1, 2.4, 0.7, "Text tokenizer\n+ embedding", "#dff5d3")

    # Unified trunk
    box(7.0, 2.6, 2.6, 2.5, "Unified causal\nTransformer\n(shared trunk)\n→ next-token CE", "#e9deff")

    # Outputs
    box(7.0, 1.0, 1.2, 0.8, "Text\nlogits", "#dff5d3")
    box(8.4, 1.0, 1.2, 0.8, "VQ token\nlogits", "#ffd9d9")

    # Arrows
    arrow(1.6, 4.85, 2.2, 5.5)         # image -> understanding
    arrow(1.6, 4.85, 2.2, 4.4)         # image -> VQ
    arrow(4.6, 5.5, 5.2, 5.5)
    arrow(4.6, 4.4, 5.2, 4.4)
    arrow(6.6, 5.5, 7.0, 4.6)
    arrow(6.6, 4.4, 7.0, 3.8)
    arrow(1.6, 1.35, 2.2, 1.45)        # text -> embed
    arrow(4.6, 1.45, 7.0, 3.0)         # embed -> trunk
    arrow(7.5, 2.6, 7.5, 1.8)
    arrow(8.9, 2.6, 8.9, 1.8)

    ax.text(5.0, 5.95, "Understanding pathway (semantic features → continuous embeddings)",
            ha='center', color='#1965b6', fontsize=8)
    ax.text(5.0, 3.85, "Generation pathway (image → discrete VQ token IDs)",
            ha='center', color='#aa1f1f', fontsize=8)
    ax.text(5.0, 0.6, "Decoupled visual encoders feed a SINGLE autoregressive Transformer",
            ha='center', fontsize=10, style='italic')

    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_architecture.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_architecture.png")


# -------- Encoder feature comparison ---------------------------------------

@torch.no_grad()
def fig_encoder_compare():
    samples = build_synthetic(n_per_combo=5, seed=42)
    real = load_real_images()
    all_imgs = stack_images(samples + real)

    vq = VQTokenizer(256, 64)
    vq.load_state_dict(torch.load(CKPT_DIR / "vq.pt", weights_only=True)); vq.eval()
    enc = UnderstandingEncoder(dim=192, depth=4, heads=4)
    siglip = torch.load(CKPT_DIR / "siglip.pt", weights_only=True)
    enc.load_state_dict(siglip["img_enc"]); enc.eval()

    # Decoupled features (CLS token)
    f_d = enc(all_imgs)[:, 0].numpy()
    # Shared (VQ encoder)
    f_s = vq.enc(all_imgs).flatten(1).numpy()

    from sklearn.decomposition import PCA
    p_d = PCA(n_components=2, random_state=0).fit_transform(f_d)
    p_s = PCA(n_components=2, random_state=0).fit_transform(f_s)

    colours = [s.colour for s in samples] + ['black', 'gray']
    shapes  = [s.shape for s in samples] + ['equation', 'meme']
    cmap = {'red':'#dc3c3c','green':'#3cc85a','blue':'#3c78f0',
            'yellow':'#f0dc46','purple':'#a05ac8','orange':'#f08c3c',
            'monochrome':'black','mixed':'gray','black':'black','gray':'gray'}
    markers = {'circle':'o','square':'s','triangle':'^','equation':'P','meme':'X'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, p, ttl in [(axes[0], p_d, "Decoupled understanding encoder\n(SigLIP-style)"),
                       (axes[1], p_s, "Shared VQ encoder\n(generation tokenizer reused)")]:
        for i in range(len(p)):
            ax.scatter(p[i, 0], p[i, 1],
                       c=cmap.get(colours[i], 'k'),
                       marker=markers.get(shapes[i], 'o'), s=60, edgecolors='k',
                       linewidths=0.4, alpha=0.85)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.grid(alpha=0.3)
    # Legend
    leg = []
    for c, hex in cmap.items():
        if c in {'monochrome','mixed','black','gray'}: continue
        leg.append(plt.Line2D([0],[0], marker='o', color='w', label=c,
                              markerfacecolor=hex, markersize=8))
    for sh, m in markers.items():
        leg.append(plt.Line2D([0],[0], marker=m, color='w', label=sh,
                              markerfacecolor='lightgray', markeredgecolor='k', markersize=8))
    fig.legend(handles=leg, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("PCA of visual features — decoupled (left) vs shared (right)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(IMG_DIR / "fig_encoder_comparison.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_encoder_comparison.png")


# -------- VQ reconstruction grid -------------------------------------------

@torch.no_grad()
def fig_vq_recon():
    samples = build_synthetic(n_per_combo=2, seed=7)
    reals = load_real_images()
    pick = samples[:6] + reals
    imgs = stack_images(pick)
    vq = VQTokenizer(256, 64)
    vq.load_state_dict(torch.load(CKPT_DIR / "vq.pt", weights_only=True)); vq.eval()
    rec, _, _, _ = vq(imgs)
    fig, axes = plt.subplots(2, len(pick), figsize=(2*len(pick), 4.2))
    for i, im in enumerate(imgs):
        a = ((im.permute(1,2,0)+1)*127.5).clamp(0,255).numpy().astype(np.uint8)
        b = ((rec[i].permute(1,2,0)+1)*127.5).clamp(0,255).numpy().astype(np.uint8)
        axes[0, i].imshow(a); axes[0, i].axis('off')
        axes[1, i].imshow(b); axes[1, i].axis('off')
        axes[0, i].set_title(pick[i].caption, fontsize=7)
    axes[0,0].set_ylabel("Original",   fontsize=10)
    axes[1,0].set_ylabel("VQ recon",   fontsize=10)
    axes[0,0].axis('on'); axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
    axes[1,0].axis('on'); axes[1,0].set_xticks([]); axes[1,0].set_yticks([])
    fig.suptitle("VQ tokenizer reconstruction (top: original, bottom: recon)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_vq_reconstruction.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_vq_reconstruction.png")


# -------- Generation grid --------------------------------------------------

def fig_generation_grid():
    npz = np.load(OUT / "generation_results.npz")
    prompts = list(npz['prompts'])
    gens_d = npz['gens_d']; gens_s = npz['gens_s']
    fig, axes = plt.subplots(2, len(prompts), figsize=(2*len(prompts), 4.4))
    for i, p in enumerate(prompts):
        axes[0, i].imshow(gens_d[i]); axes[0, i].axis('off')
        axes[1, i].imshow(gens_s[i]); axes[1, i].axis('off')
        axes[0, i].set_title(p, fontsize=8)
    axes[0,0].text(-0.15, 0.5, "Decoupled", rotation=90, transform=axes[0,0].transAxes,
                   ha='right', va='center', fontsize=10)
    axes[1,0].text(-0.15, 0.5, "Shared",    rotation=90, transform=axes[1,0].transAxes,
                   ha='right', va='center', fontsize=10)
    fig.suptitle("Text-to-image autoregressive generation (greedy decoded VQ tokens)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_generation_grid.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_generation_grid.png")


# -------- Understanding qualitative ----------------------------------------

def fig_understanding_qualitative():
    summary = json.load(open(OUT / "results_summary.json"))
    real = load_real_images()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, samp in zip(axes, real):
        ax.imshow(samp.image.transpose(1, 2, 0))
        ax.axis('off')
        d_cap = summary['real_captions_decoupled'].get(samp.caption, "[N/A]")
        s_cap = summary['real_captions_shared'].get(samp.caption, "[N/A]")
        title = (f"Reference probe caption: \"{samp.caption}\"\n"
                 f"Decoupled trunk says: \"{d_cap}\"\n"
                 f"Shared    trunk says: \"{s_cap}\"")
        ax.set_title(title, fontsize=8)
    fig.suptitle("Qualitative VQA / image-conditioned text generation on real images",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_understanding_qualitative.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_understanding_qualitative.png")


# -------- Ablation bar chart -----------------------------------------------

def fig_ablation():
    summary = json.load(open(OUT / "results_summary.json"))
    p = summary['linear_probe']
    c = summary['captioning_test']
    labels = ["Probe-shape", "Probe-colour", "Caption-shape", "Caption-colour"]
    decoupled = [p['decoupled_shape_acc'], p['decoupled_colour_acc'],
                 c['decoupled']['shape_acc'], c['decoupled']['colour_acc']]
    shared = [p['shared_vq_shape_acc'], p['shared_vq_colour_acc'],
              c['shared']['shape_acc'], c['shared']['colour_acc']]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, decoupled, w, label='Decoupled (semantic + VQ)', color='#3c78f0')
    ax.bar(x + w/2, shared,    w, label='Shared (single VQ encoder)', color='#dc3c3c')
    for i, v in enumerate(decoupled):
        ax.text(i - w/2, v + 0.015, f"{v:.2f}", ha='center', fontsize=8)
    for i, v in enumerate(shared):
        ax.text(i + w/2, v + 0.015, f"{v:.2f}", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Ablation: decoupled vs shared visual encoder")
    ax.grid(alpha=0.3, axis='y'); ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_ablation_table.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_ablation_table.png")


# -------- Saliency map -----------------------------------------------------

@torch.no_grad()
def fig_saliency():
    enc = UnderstandingEncoder(dim=192, depth=4, heads=4)
    siglip = torch.load(CKPT_DIR / "siglip.pt", weights_only=True)
    enc.load_state_dict(siglip["img_enc"]); enc.eval()
    real = load_real_images()
    imgs = stack_images(real)
    feats = enc(imgs)  # (B, 17, dim) ; CLS at 0
    cls = feats[:, 0:1]      # (B, 1, D)
    patches = feats[:, 1:]   # (B, 16, D)
    sim = (patches * cls).sum(-1)  # (B, 16) cosine-like
    sim = (sim - sim.amin(1, keepdim=True)) / (sim.amax(1, keepdim=True) - sim.amin(1, keepdim=True) + 1e-6)
    sim = sim.view(-1, 4, 4).numpy()  # 4x4 patch grid

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    for i, samp in enumerate(real):
        axes[2*i].imshow(samp.image.transpose(1, 2, 0))
        axes[2*i].set_title(samp.caption, fontsize=9); axes[2*i].axis('off')
        # Upsample 4x4 saliency to 64x64
        from scipy.ndimage import zoom
        sm = zoom(sim[i], 16, order=1)
        axes[2*i+1].imshow(samp.image.transpose(1, 2, 0))
        axes[2*i+1].imshow(sm, alpha=0.5, cmap='jet')
        axes[2*i+1].set_title("CLS-patch attention", fontsize=9)
        axes[2*i+1].axis('off')
    fig.suptitle("Understanding-encoder saliency (CLS↔patch similarity)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_understanding_saliency.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("saved fig_understanding_saliency.png")


def main():
    fig_architecture()
    fig_encoder_compare()
    fig_vq_recon()
    fig_generation_grid()
    fig_understanding_qualitative()
    fig_ablation()
    fig_saliency()


if __name__ == "__main__":
    main()
