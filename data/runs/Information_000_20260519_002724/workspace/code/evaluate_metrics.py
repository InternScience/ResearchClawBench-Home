"""
Compute proper evaluation metrics.
"""
import sys
sys.path.insert(0, 'code')
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from evaluate_and_plot import load_unified, test_ds, vqvae, generate_image_from_text
from train_unified_only import collate_generation, collate_understanding
from tokenizer import decode, PAD_ID

# Metrics
def token_accuracy(preds, targets):
    mask = targets != PAD_ID
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def char_similarity(pred_str, tgt_str):
    # simple char-level accuracy
    l = max(len(pred_str), len(tgt_str))
    if l == 0:
        return 1.0
    matches = sum(1 for a, b in zip(pred_str, tgt_str) if a == b)
    return matches / l

@torch.no_grad()
def evaluate_understanding(transformer, vis_enc, max_batches=10):
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=lambda x: x)
    total_tok_acc = 0.0
    total_char_sim = 0.0
    n_batches = 0
    samples = []
    for batch in loader:
        imgs, seqs = collate_understanding(batch)
        imgs = imgs.to('cpu')
        seqs = seqs.to('cpu')
        inp = seqs[:, :-1]
        tgt = seqs[:, 1:]
        vis = vis_enc(imgs)
        logits = transformer(inp, continuous_prefix=vis)
        logits = logits[:, vis.size(1):, :]
        preds = logits.argmax(dim=-1)
        total_tok_acc += token_accuracy(preds, tgt)
        for i in range(tgt.size(0)):
            tgt_str = decode(tgt[i].tolist())
            pred_str = decode(preds[i].tolist()[:tgt.size(1)])
            total_char_sim += char_similarity(pred_str, tgt_str)
            if len(samples) < 5:
                samples.append((batch[i][1]['caption'], pred_str, tgt_str))
        n_batches += 1
        if n_batches >= max_batches:
            break
    n_samples = n_batches * 64
    return total_tok_acc / n_batches, total_char_sim / n_samples, samples

@torch.no_grad()
def evaluate_generation(transformer, max_samples=20):
    mse_list = []
    samples = []
    for i in range(max_samples):
        img, meta = test_ds[i]
        text = meta['caption']
        gen_img = generate_image_from_text(text, transformer, vqvae)
        if gen_img is None:
            continue
        mse = F.mse_loss(gen_img, img).item()
        mse_list.append(mse)
        if len(samples) < 5:
            samples.append((text, gen_img, img))
    return np.mean(mse_list) if mse_list else 0.0, samples

print("Evaluating Decoupled...")
trans_dec, vis_dec, _ = load_unified('outputs/unified_decoupled.pt')
under_tok_dec, under_char_dec, under_samp_dec = evaluate_understanding(trans_dec, vis_dec)
gen_mse_dec, gen_samp_dec = evaluate_generation(trans_dec)

print("Evaluating Coupled...")
trans_coup, vis_coup, _ = load_unified('outputs/unified_coupled.pt')
under_tok_coup, under_char_coup, under_samp_coup = evaluate_understanding(trans_coup, vis_coup)
gen_mse_coup, gen_samp_coup = evaluate_generation(trans_coup)

metrics = {
    'decoupled': {
        'under_token_acc': under_tok_dec,
        'under_char_sim': under_char_dec,
        'gen_mse': gen_mse_dec,
    },
    'coupled': {
        'under_token_acc': under_tok_coup,
        'under_char_sim': under_char_coup,
        'gen_mse': gen_mse_coup,
    }
}
print(metrics)
with open('outputs/eval_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save sample texts for report
with open('outputs/understanding_samples.json', 'w') as f:
    json.dump({'decoupled': under_samp_dec, 'coupled': under_samp_coup}, f, indent=2)
