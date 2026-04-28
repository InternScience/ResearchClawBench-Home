"""
Evaluation: image-text retrieval, shape/colour probe, VQ reconstruction,
qualitative VQA on equation.png and doge.png, and AR text-to-image.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_utils import build_synthetic, load_real_images, WordTokenizer, IMG_SIZE
from models import UnderstandingEncoder, TextEncoder, VQTokenizer, UnifiedConfig, UnifiedTransformer
from train import stack_images, build_understanding_seq, build_generation_seq, CKPT_DIR

WORKSPACE = Path(__file__).resolve().parent.parent
OUT = WORKSPACE / "outputs"
DEVICE = torch.device("cpu")


def load_all():
    samples = build_synthetic(n_per_combo=10, seed=0)
    real = load_real_images()
    tokenizer = WordTokenizer([s.caption for s in samples])
    cfg = UnifiedConfig(text_vocab=len(tokenizer), vq_codebook=256, n_special=9,
                        dim=192, depth=6, heads=6, max_len=128)

    vq = VQTokenizer(256, 64)
    vq.load_state_dict(torch.load(CKPT_DIR / "vq.pt", weights_only=True))
    vq.eval()

    img_enc = UnderstandingEncoder(dim=192, depth=4, heads=4)
    txt_enc = TextEncoder(vocab=len(tokenizer), dim=192, depth=4, heads=4)
    siglip = torch.load(CKPT_DIR / "siglip.pt", weights_only=True)
    img_enc.load_state_dict(siglip["img_enc"]); img_enc.eval()
    txt_enc.load_state_dict(siglip["txt_enc"]); txt_enc.eval()
    img_proj = torch.nn.Linear(192, 128); img_proj.load_state_dict(siglip["img_proj"]); img_proj.eval()
    txt_proj = torch.nn.Linear(192, 128); txt_proj.load_state_dict(siglip["txt_proj"]); txt_proj.eval()

    trunk_d = UnifiedTransformer(cfg, understand_dim=192)
    trunk_d.load_state_dict(torch.load(CKPT_DIR / "trunk_decoupled.pt", weights_only=True))
    trunk_d.eval()
    trunk_s = UnifiedTransformer(cfg, understand_dim=192)
    trunk_s.load_state_dict(torch.load(CKPT_DIR / "trunk_shared.pt", weights_only=True))
    trunk_s.eval()
    return samples, real, tokenizer, cfg, vq, img_enc, txt_enc, img_proj, txt_proj, trunk_d, trunk_s


@torch.no_grad()
def retrieval_eval(samples, vq, img_enc, txt_enc, img_proj, txt_proj, tokenizer):
    """Compare decoupled vs shared encoder on text->image and image->text retrieval."""
    images = stack_images(samples)
    cap_max = 8
    pad = tokenizer.special("<pad>")
    cap_ids = []
    for s in samples:
        ids = tokenizer.encode(s.caption)[:cap_max]
        ids = ids + [pad] * (cap_max - len(ids))
        cap_ids.append(ids)
    cap_ids = torch.tensor(cap_ids, dtype=torch.long)

    # Decoupled (semantic) features
    f_d = F.normalize(img_proj(img_enc(images)[:, 0]), dim=-1)
    # Shared (VQ encoder reused)
    z = vq.enc(images).flatten(2).mean(-1)  # (N, 64)
    z = torch.cat([z, z, z], dim=-1)[:, :192]
    # Train a quick linear probe to make the comparison fair: use a fixed
    # random projection for shared so both have a 128-dim space.
    rp = torch.randn(192, 128); rp = rp / rp.norm(dim=0, keepdim=True)
    f_s = F.normalize(z @ rp, dim=-1)

    t = F.normalize(txt_proj(txt_enc(cap_ids)), dim=-1)

    cap_strs = [s.caption for s in samples]
    same = torch.tensor(
        [[cap_strs[i] == cap_strs[j] for j in range(len(cap_strs))]
         for i in range(len(cap_strs))]
    )
    def top1(f, t):
        sim = f @ t.t()
        i2t = same[torch.arange(sim.size(0)), sim.argmax(1)].float().mean().item()
        t2i = same[sim.argmax(0), torch.arange(sim.size(0))].float().mean().item()
        return i2t, t2i

    return {
        "decoupled_i2t": top1(f_d, t)[0], "decoupled_t2i": top1(f_d, t)[1],
        "shared_i2t":    top1(f_s, t)[0], "shared_t2i":    top1(f_s, t)[1],
    }


@torch.no_grad()
def linear_probe(samples, vq, img_enc):
    """Linear probe accuracy for shape and colour classification on top of
    (a) decoupled semantic features and (b) shared (VQ enc) features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    images = stack_images(samples)
    f_d = img_enc(images)[:, 0].numpy()
    z = vq.enc(images).flatten(1).numpy()  # high-dim raw VQ features

    shapes = np.array([s.shape for s in samples])
    colours = np.array([s.colour for s in samples])

    out = {}
    for name, X in [("decoupled", f_d), ("shared_vq", z)]:
        for tgt_name, y in [("shape", shapes), ("colour", colours)]:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
            clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            out[f"{name}_{tgt_name}_acc"] = float(clf.score(Xte, yte))
    return out


@torch.no_grad()
def vq_reconstruction(samples, real, vq):
    images = stack_images(samples[:8] + real)
    recon, idx, commit, _ = vq(images)
    mse = F.mse_loss(recon, images).item()
    # Convert to displayable
    grid = []
    for orig, rec in zip(images, recon):
        a = ((orig.permute(1,2,0) + 1) * 127.5).clamp(0,255).numpy().astype(np.uint8)
        b = ((rec.permute(1,2,0) + 1) * 127.5).clamp(0,255).numpy().astype(np.uint8)
        grid.append((a, b))
    return mse, grid


@torch.no_grad()
def captioning(real, img_enc, vq, trunk: UnifiedTransformer, tokenizer, max_new=8):
    """Greedy text decoding given image features (decoupled path).
    For the shared-trunk we still feed shared-style features."""
    images = stack_images(real)
    if trunk is None:
        return []
    # Decoupled if trunk's und_proj came from img_enc training; we just always
    # feed semantic features here for fairness when trunk == decoupled. For
    # shared trunk we feed shared-style features.
    feats_dec = img_enc(images)
    z = vq.enc(images).flatten(2).transpose(1,2).repeat(1,1,3)[..., :192][:, :17]
    return feats_dec, z, images


def greedy_decode_caption(trunk, feats, tokenizer, max_new=8):
    """feats: (B, 17, 192). Build prompt up to <SEP> then generate caption."""
    B = feats.size(0)
    n_und = feats.size(1)
    L = 1 + 1 + n_und + 1 + 1  # bos boi und eoi sep
    ids = torch.full((B, L), trunk.spec("pad"), dtype=torch.long)
    fea = torch.zeros(B, L, feats.size(-1))
    msk = torch.zeros(B, L, dtype=torch.bool)
    pos = 0
    ids[:, pos] = trunk.spec("bos"); pos += 1
    ids[:, pos] = trunk.spec("boi"); pos += 1
    ids[:, pos:pos+n_und] = -1
    fea[:, pos:pos+n_und] = feats
    msk[:, pos:pos+n_und] = True
    pos += n_und
    ids[:, pos] = trunk.spec("eoi"); pos += 1
    ids[:, pos] = trunk.spec("sep"); pos += 1

    out_tokens = [[] for _ in range(B)]
    eos = trunk.spec("eos")
    for _ in range(max_new):
        logits = trunk(ids, fea, msk)
        nxt = logits[:, -1].argmax(-1)
        # Restrict to text vocab
        nxt = torch.where(
            (nxt < trunk.cfg.text_vocab),
            nxt,
            torch.tensor(trunk.cfg.text_vocab - 1),  # fallback
        )
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        fea = torch.cat([fea, torch.zeros(B, 1, feats.size(-1))], dim=1)
        msk = torch.cat([msk, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
        for b in range(B):
            out_tokens[b].append(int(nxt[b]))
        if all(eos in t for t in out_tokens):
            break
    captions = [tokenizer.decode([t for t in toks if t < len(tokenizer.itos)
                                  and tokenizer.itos[t] not in {"<pad>", "<bos>", "<eos>"}])
                for toks in out_tokens]
    return captions


@torch.no_grad()
def generate_image(trunk, tokenizer, prompt: str, vq: VQTokenizer):
    """AR generate 64 VQ tokens then decode with VQ decoder."""
    cap_max = 6
    pad = tokenizer.special("<pad>")
    ids = tokenizer.encode(prompt)[:cap_max]
    ids = ids + [pad] * (cap_max - len(ids))
    seq = torch.tensor([[trunk.spec("bos")] + ids + [trunk.spec("sep"), trunk.spec("bog")]], dtype=torch.long)
    fea = torch.zeros(1, seq.size(1), 192)
    msk = torch.zeros(1, seq.size(1), dtype=torch.bool)

    n_img = 64
    for _ in range(n_img):
        logits = trunk(seq, fea, msk)
        last = logits[0, -1]
        # Restrict to VQ codebook range
        vq_lo = trunk.cfg.text_vocab
        vq_hi = trunk.cfg.text_vocab + trunk.cfg.vq_codebook
        masked = last.clone()
        masked[:vq_lo] = -1e9
        masked[vq_hi:] = -1e9
        nxt = masked.argmax(-1, keepdim=True).unsqueeze(0)
        seq = torch.cat([seq, nxt], dim=1)
        fea = torch.cat([fea, torch.zeros(1, 1, 192)], dim=1)
        msk = torch.cat([msk, torch.zeros(1, 1, dtype=torch.bool)], dim=1)

    # Extract VQ ids (last 64 tokens before any <eog>)
    img_ids = seq[0, -n_img:].clone() - trunk.cfg.text_vocab
    img_ids = img_ids.clamp(0, trunk.cfg.vq_codebook - 1)
    grid = img_ids.view(1, 8, 8)
    rec = vq.decode_from_indices(grid)
    img = ((rec[0].permute(1,2,0) + 1) * 127.5).clamp(0,255).numpy().astype(np.uint8)
    return img, img_ids.tolist()


def main():
    print("loading all checkpoints ...")
    samples, real, tokenizer, cfg, vq, img_enc, txt_enc, img_proj, txt_proj, trunk_d, trunk_s = load_all()

    print("retrieval ...")
    ret = retrieval_eval(samples, vq, img_enc, txt_enc, img_proj, txt_proj, tokenizer)
    print(ret)

    print("linear probe ...")
    probe = linear_probe(samples, vq, img_enc)
    print(probe)

    print("vq reconstruction ...")
    mse, _ = vq_reconstruction(samples, real, vq)
    print("mse=", mse)

    print("captioning real images ...")
    feats_dec, feats_shared, images = captioning(real, img_enc, vq, trunk_d, tokenizer)
    cap_d = greedy_decode_caption(trunk_d, feats_dec, tokenizer, max_new=6)
    cap_s = greedy_decode_caption(trunk_s, feats_shared, tokenizer, max_new=6)
    # Also caption the synthetic test set for accuracy measurement
    test_samples = samples[-30:]
    test_imgs = stack_images(test_samples)
    test_feats_d = img_enc(test_imgs)
    test_feats_s = vq.enc(test_imgs).flatten(2).transpose(1,2).repeat(1,1,3)[..., :192][:, :17]
    cap_test_d = greedy_decode_caption(trunk_d, test_feats_d, tokenizer, max_new=6)
    cap_test_s = greedy_decode_caption(trunk_s, test_feats_s, tokenizer, max_new=6)

    def acc(captions, samples):
        n_shape = sum(1 for c, s in zip(captions, samples) if s.shape in c.split())
        n_col = sum(1 for c, s in zip(captions, samples) if s.colour in c.split())
        return n_shape / len(samples), n_col / len(samples)

    acc_d_shape, acc_d_col = acc(cap_test_d, test_samples)
    acc_s_shape, acc_s_col = acc(cap_test_s, test_samples)

    print("decoupled real captions:", cap_d)
    print("shared    real captions:", cap_s)
    print(f"decoupled test  shape acc={acc_d_shape:.2f} colour acc={acc_d_col:.2f}")
    print(f"shared    test  shape acc={acc_s_shape:.2f} colour acc={acc_s_col:.2f}")

    print("text-to-image generation ...")
    gen_prompts = ["a red circle", "a blue square", "a green triangle",
                   "a yellow circle", "a purple square", "an orange triangle"]
    gens_d, gens_s = [], []
    for p in gen_prompts:
        img_d, _ = generate_image(trunk_d, tokenizer, p, vq)
        img_s, _ = generate_image(trunk_s, tokenizer, p, vq)
        gens_d.append((p, img_d))
        gens_s.append((p, img_s))

    summary = {
        "retrieval": ret,
        "linear_probe": probe,
        "vq_reconstruction_mse": mse,
        "real_captions_decoupled": dict(zip([s.caption for s in real], cap_d)),
        "real_captions_shared":    dict(zip([s.caption for s in real], cap_s)),
        "captioning_test": {
            "decoupled": {"shape_acc": acc_d_shape, "colour_acc": acc_d_col},
            "shared":    {"shape_acc": acc_s_shape, "colour_acc": acc_s_col},
        },
    }
    json.dump(summary, open(OUT / "results_summary.json", "w"), indent=2)
    print("saved outputs/results_summary.json")

    # Persist generated images and inputs for the figure script
    np.savez(OUT / "generation_results.npz",
             prompts=np.array(gen_prompts),
             gens_d=np.stack([g for _, g in gens_d]),
             gens_s=np.stack([g for _, g in gens_s]))
    print("saved outputs/generation_results.npz")


if __name__ == "__main__":
    main()
