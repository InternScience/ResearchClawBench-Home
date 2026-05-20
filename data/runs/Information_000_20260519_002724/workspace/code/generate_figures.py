"""
Generate figures for the report using saved metrics and models.
"""
import os
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

import sys
sys.path.insert(0, 'code')
from vqvae import VQVAE
from unified_model import UnifiedTransformer, UnderstandingEncoder, CoupledEncoder
from tokenizer import encode_raw, decode, PAD_ID, SOS_ID, EOS_ID, IMG_START_ID, IMG_END_ID, IMG_TOKEN_START, TOTAL_VOCAB
from train_unified_only import SyntheticDataset

DEVICE = torch.device('cpu')
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
D_FF = 256

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs('report/images', exist_ok=True)

# Load models
vqvae = VQVAE().to(DEVICE)
vqvae.load_state_dict(torch.load('outputs/vqvae_best.pt', map_location=DEVICE))
vqvae.eval()

def load_unified(path):
    ckpt = torch.load(path, map_location=DEVICE)
    transformer = UnifiedTransformer(vocab_size=TOTAL_VOCAB, d_model=D_MODEL,
                                     n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF).to(DEVICE)
    transformer.load_state_dict(ckpt['transformer'], strict=False)
    use_decoupled = ckpt['use_decoupled']
    if use_decoupled:
        vis_enc = UnderstandingEncoder(d_model=D_MODEL).to(DEVICE)
    else:
        vis_enc = CoupledEncoder(vqvae.encoder, vqvae.quantizer, d_model=D_MODEL).to(DEVICE)
    vis_enc.load_state_dict(ckpt['vis_enc'])
    transformer.eval()
    vis_enc.eval()
    return transformer, vis_enc

transformer_dec, vis_enc_dec = load_unified('outputs/unified_decoupled.pt')
transformer_coup, vis_enc_coup = load_unified('outputs/unified_coupled.pt')

test_ds = SyntheticDataset('outputs/synthetic_test')

# Load metrics
with open('outputs/eval_metrics.json') as f:
    metrics = json.load(f)

# ------------------- Helper Functions -------------------
@torch.no_grad()
def generate_image_from_text(text, transformer, vqvae, max_len=100):
    text_ids = encode_raw(text, max_len=20)
    seq = [SOS_ID] + text_ids + [IMG_START_ID]
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    for _ in range(64):
        logits = transformer(seq_tensor)
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        probs[0, :IMG_TOKEN_START] = 0
        probs[0, IMG_TOKEN_START + 256:] = 0
        next_token = torch.multinomial(probs, 1).item()
        if next_token < IMG_TOKEN_START or next_token >= IMG_TOKEN_START + 256:
            next_token = IMG_TOKEN_START
        if next_token == IMG_END_ID or next_token == EOS_ID:
            break
        seq.append(next_token)
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    try:
        img_start_idx = seq.index(IMG_START_ID)
    except ValueError:
        return None
    img_tokens = seq[img_start_idx+1:]
    img_tokens = [t for t in img_tokens if t != IMG_END_ID and t != EOS_ID and t != PAD_ID]
    if len(img_tokens) < 64:
        img_tokens = img_tokens + [IMG_TOKEN_START] * (64 - len(img_tokens))
    img_tokens = img_tokens[:64]
    indices = torch.tensor([t - IMG_TOKEN_START for t in img_tokens], dtype=torch.long).view(1, 8, 8).to(DEVICE)
    z_q = vqvae.quantizer.embeddings(indices).permute(0, 3, 1, 2)
    img = vqvae.decode(z_q)
    img = img.squeeze(0).cpu().clamp(0, 1)
    return img

@torch.no_grad()
def answer_question(img, question, transformer, vis_enc):
    img = img.unsqueeze(0).to(DEVICE)
    q_ids = encode_raw(question, max_len=20)
    seq = [SOS_ID] + q_ids
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    vis = vis_enc(img)
    for _ in range(20):
        logits = transformer(seq_tensor, continuous_prefix=vis)
        logits = logits[:, vis.size(1):, :]
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        probs[0, IMG_START_ID:IMG_TOKEN_START+256] = 0
        next_token = torch.multinomial(probs, 1).item()
        if next_token == EOS_ID:
            break
        seq.append(next_token)
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    return decode(seq, skip_specials=True)

# ------------------- Figures -------------------

# 1. Architecture Diagram
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

def draw_box(x, y, w, h, text, color='lightblue'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, wrap=True)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

draw_box(0.2, 3.5, 1.2, 0.8, 'Image Input', 'lightgray')
draw_box(2.0, 4.5, 1.8, 0.8, 'Understanding Encoder\n(Continuous CNN)', 'lightgreen')
draw_arrow(1.4, 3.9, 2.0, 4.7)
draw_box(2.0, 2.2, 1.8, 0.8, 'Generation Encoder\n(VQ-VAE Tokenizer)', 'lightsalmon')
draw_arrow(1.4, 3.9, 2.0, 2.8)
draw_box(4.5, 3.0, 2.0, 1.2, 'Unified Transformer\n(Autoregressive)', 'lightyellow')
draw_arrow(3.8, 4.7, 4.5, 3.8)
draw_arrow(3.8, 2.8, 4.5, 3.4)
draw_box(4.5, 5.2, 1.5, 0.6, 'Text Input', 'lightgray')
draw_arrow(5.25, 5.2, 5.25, 4.2)
draw_box(7.0, 4.5, 1.5, 0.8, 'Text Output\n(VQA / Caption)', 'lightgreen')
draw_arrow(6.5, 3.8, 7.0, 4.7)
draw_box(7.0, 2.0, 1.5, 0.8, 'Image Tokens\n(Decoder)', 'lightsalmon')
draw_arrow(6.5, 3.4, 7.0, 2.6)
draw_box(9.0, 2.0, 1.2, 0.8, 'Image Output', 'lightgray')
draw_arrow(8.5, 2.4, 9.0, 2.4)
ax.set_title('Unified Autoregressive Framework with Decoupled Visual Encoding', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/architecture.png', dpi=150)
plt.close()

# 2. VQ-VAE Reconstructions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    img, meta = test_ds[i]
    with torch.no_grad():
        recon, _, _, _, _ = vqvae(img.unsqueeze(0).to(DEVICE))
    recon = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_np = img.permute(1, 2, 0).numpy()
    axes[0, i].imshow(img_np)
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon)
    axes[1, i].set_title('Recon')
    axes[1, i].axis('off')
plt.suptitle('VQ-VAE Reconstructions on Synthetic Test Set', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/vqvae_reconstruction.png', dpi=150)
plt.close()

# 3. Generated Images (Decoupled)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    img, meta = test_ds[i]
    text = meta['caption']
    gen_img = generate_image_from_text(text, transformer_dec, vqvae)
    axes[0, i].imshow(gen_img.permute(1, 2, 0).numpy())
    axes[0, i].set_title(f'Gen: {text}')
    axes[0, i].axis('off')
    axes[1, i].imshow(img.permute(1, 2, 0).numpy())
    axes[1, i].set_title('GT')
    axes[1, i].axis('off')
plt.suptitle('Text-to-Image Generation (Decoupled Model)', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/generated_images_decoupled.png', dpi=150)
plt.close()

# 4. Understanding Samples (Decoupled)
fig, axes = plt.subplots(2, 5, figsize=(12, 4))
with open('outputs/understanding_samples.json') as f:
    under_data = json.load(f)
for i, (caption, pred, tgt) in enumerate(under_data['decoupled']):
    img, _ = test_ds[i]
    axes[0, i].imshow(img.permute(1, 2, 0).numpy())
    axes[0, i].set_title(f'{caption}')
    axes[0, i].axis('off')
    axes[1, i].text(0.5, 0.5, f'Pred: {pred}\nTrue: {tgt}', ha='center', va='center', fontsize=9)
    axes[1, i].axis('off')
plt.suptitle('Visual Question Answering (Decoupled Model)', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/understanding_samples.png', dpi=150)
plt.close()

# 5. Training Curves
with open('outputs/vqvae_history.json') as f:
    vq_hist = json.load(f)
with open('outputs/unified_decoupled_history.json') as f:
    dec_hist = json.load(f)
with open('outputs/unified_coupled_history.json') as f:
    coup_hist = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(vq_hist['train'], label='Train')
axes[0].plot(vq_hist['val'], label='Val')
axes[0].set_title('VQ-VAE Training')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(dec_hist['gen_train'], label='Dec Gen Train')
axes[1].plot(dec_hist['gen_val'], label='Dec Gen Val')
axes[1].plot(coup_hist['gen_train'], label='Coup Gen Train', linestyle='--')
axes[1].plot(coup_hist['gen_val'], label='Coup Gen Val', linestyle='--')
axes[1].set_title('Generation Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(dec_hist['under_train'], label='Dec Under Train')
axes[2].plot(dec_hist['under_val'], label='Dec Under Val')
axes[2].plot(coup_hist['under_train'], label='Coup Under Train', linestyle='--')
axes[2].plot(coup_hist['under_val'], label='Coup Under Val', linestyle='--')
axes[2].set_title('Understanding Loss')
axes[2].set_xlabel('Epoch')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('report/images/training_curves.png', dpi=150)
plt.close()

# 6. Ablation Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
models = ['Decoupled', 'Coupled']
acc = [metrics['decoupled']['under_token_acc'], metrics['coupled']['under_token_acc']]
bars = axes[0].bar(models, acc, color=['seagreen', 'coral'])
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel('Token-level Accuracy')
axes[0].set_title('Understanding Accuracy')
for bar, val in zip(bars, acc):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2%}', ha='center', va='bottom')

mse = [metrics['decoupled']['gen_mse'], metrics['coupled']['gen_mse']]
bars = axes[1].bar(models, mse, color=['seagreen', 'coral'])
axes[1].set_ylabel('MSE')
axes[1].set_title('Generation Quality (MSE)')
for bar, val in zip(bars, mse):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('report/images/ablation.png', dpi=150)
plt.close()

# 7. Real Image Results Figure
real_results = {}
for name, transformer, vis_enc in [('Decoupled', transformer_dec, vis_enc_dec), ('Coupled', transformer_coup, vis_enc_coup)]:
    real_results[name] = {
        'equation': answer_question(
            torch.from_numpy(np.array(Image.open('data/equation.png').convert('RGB').resize((32,32)))).permute(2,0,1).float()/255.0,
            "What does the image show?", transformer, vis_enc),
        'doge': answer_question(
            torch.from_numpy(np.array(Image.open('data/doge.png').convert('RGB').resize((32,32)))).permute(2,0,1).float()/255.0,
            "Describe the image.", transformer, vis_enc),
    }
with open('outputs/real_results.json', 'w') as f:
    json.dump(real_results, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(Image.open('data/equation.png'))
axes[0].set_title(f"Equation\nDec: {real_results['Decoupled']['equation']}\nCoup: {real_results['Coupled']['equation']}", fontsize=9)
axes[0].axis('off')
axes[1].imshow(Image.open('data/doge.png'))
axes[1].set_title(f"Doge\nDec: {real_results['Decoupled']['doge']}\nCoup: {real_results['Coupled']['doge']}", fontsize=9)
axes[1].axis('off')
plt.suptitle('Qualitative Results on Real Images', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/real_image_results.png', dpi=150)
plt.close()

print("All figures regenerated.")
