"""
Plot the framework architecture as a publication-style block diagram (matplotlib).
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG  = os.path.join(ROOT, "report", "images")
os.makedirs(IMG, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")

def box(x, y, w, h, text, color, fc=None, fontsize=10, ec=None):
    fc = fc or color
    ec = ec or "#222"
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            wrap=True)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="->", mutation_scale=18, color="#333", lw=1.4)
    ax.add_patch(a)

# Inputs (left column)
box(0.2, 5.6, 2.7, 0.8, "Protein\nsequence (1-letter)", "white", fc="#cfe2f3", fontsize=10)
box(0.2, 4.4, 2.7, 0.8, "Nucleic-acid\nsequence (A/C/G/U/T)", "white", fc="#cfe2f3", fontsize=10)
box(0.2, 3.2, 2.7, 0.8, "Ligand graph\n(SDF: atoms + bonds)", "white", fc="#cfe2f3", fontsize=10)

# Tokenizer
box(3.4, 4.4, 2.4, 0.9, "Tokenizer\n(token type + id)", "white", fc="#fff2cc", fontsize=11)
arrow(2.9, 6.0, 3.4, 5.0); arrow(2.9, 4.8, 3.4, 4.85); arrow(2.9, 3.6, 3.4, 4.7)

# Token & pair embedding
box(3.4, 2.9, 2.4, 0.9, "Token + pair\nembedding\n(s∈ℝᴺˣᵈ, z∈ℝᴺˣᴺˣᵈᶻ)",
    "white", fc="#ffe599", fontsize=10)
arrow(4.6, 4.4, 4.6, 3.8)

# Pairformer-lite trunk
box(6.3, 3.5, 3.0, 1.5,
    "Pairformer-lite trunk\n• triangle multiplicative\n  (low-rank)\n"
    "• pair self-attention\n• outer-product-mean", "white",
    fc="#d9ead3", fontsize=10)
arrow(5.8, 3.4, 6.3, 4.0)
arrow(5.8, 4.7, 6.3, 4.5)

# Diffusion module
box(9.7, 2.9, 3.0, 2.1,
    "Diffusion module\n(ε-prediction)\n\nx_t, s, z, t  →  ε̂\nDDPM cosine schedule",
    "white", fc="#f4cccc", fontsize=10)
arrow(9.3, 4.2, 9.7, 4.0)

# Sampler / output
box(9.7, 0.9, 3.0, 1.4,
    "Reverse sampler\nx_T → x_0\n(predicted 3-D coords)",
    "white", fc="#f4cccc", fontsize=10)
arrow(11.2, 2.85, 11.2, 2.35)

# Output
box(0.2, 0.4, 6.0, 1.4,
    "Predicted complex structure\n(protein backbone Cα + ligand heavy-atom coords)\n"
    "Cα-RMSD (Kabsch) | Hungarian-matched ligand RMSD",
    "white", fc="#cccccc", fontsize=10)
arrow(9.7, 1.6, 6.2, 1.1)

# Legend strip
ax.text(6.5, 6.8, "Unified AlphaFold-3-style diffusion framework for biomolecular complexes",
        ha="center", va="center", fontsize=14, weight="bold")

plt.savefig(os.path.join(IMG, "framework_architecture.png"), dpi=160,
            bbox_inches="tight")
plt.close()
print("Saved framework_architecture.png")
