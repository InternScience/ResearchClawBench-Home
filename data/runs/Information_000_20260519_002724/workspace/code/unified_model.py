"""
Unified autoregressive model with decoupled visual encoding.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', None)

    def forward(self, x):
        B, T, C = x.size()
        if self.mask is None or self.mask.size(0) < T:
            self.mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class UnifiedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, d_ff=512, max_len=512, dropout=0.1, img_token_start=100, num_img_tokens=256):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.img_token_start = img_token_start
        self.num_img_tokens = num_img_tokens
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, continuous_prefix=None):
        # x: (B, T) token ids
        # continuous_prefix: (B, N, d_model) optional continuous embeddings prepended
        B, T = x.size()
        h = self.token_embed(x)
        if continuous_prefix is not None:
            h = torch.cat([continuous_prefix, h], dim=1)
            T = h.size(1)
        h = self.pos_enc(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


class UnderstandingEncoder(nn.Module):
    """Small CNN encoder for understanding (continuous features)."""
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8->4
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.net(x).view(x.size(0), -1)  # (B, 64)
        return self.proj(feat).unsqueeze(1)  # (B, 1, d_model)


class CoupledEncoder(nn.Module):
    """Uses VQ-VAE encoder as the single visual encoder (coupled baseline)."""
    def __init__(self, vqvae_encoder, quantizer, d_model=256):
        super().__init__()
        self.encoder = vqvae_encoder
        self.quantizer = quantizer
        self.embed = nn.Embedding(quantizer.num_embeddings, d_model)

    def forward(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.quantizer(z)
        # discrete tokens -> embeddings
        B, H, W = indices.shape
        tok = indices.view(B, H * W)
        emb = self.embed(tok)  # (B, H*W, d_model)
        return emb


if __name__ == '__main__':
    model = UnifiedTransformer(vocab_size=356)
    x = torch.randint(0, 356, (2, 20))
    logits = model(x)
    print("Logits:", logits.shape)

    enc = UnderstandingEncoder()
    img = torch.randn(2, 3, 32, 32)
    vis = enc(img)
    print("Visual prefix:", vis.shape)

    logits2 = model(x, continuous_prefix=vis)
    print("Logits with prefix:", logits2.shape)
