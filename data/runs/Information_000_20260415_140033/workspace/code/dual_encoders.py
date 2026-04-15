"""
Dual Visual Encoders for Understanding and Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnderstandingEncoder(nn.Module):
    """SigLIP-style encoder for visual understanding tasks"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, n_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, 
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.projection(x[:, 0])
        return F.normalize(x, dim=-1)


class GenerationEncoder(nn.Module):
    """VQVAE-style encoder for image generation tasks"""
    def __init__(self, in_chans=3, embed_dim=256, n_layers=4, codebook_size=16384, latent_dim=8):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        
        # Encoder
        layers = []
        in_ch = in_chans
        for i in range(n_layers):
            out_ch = embed_dim * (2 ** i)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            ])
            in_ch = out_ch
        
        layers.append(nn.Conv2d(in_ch, latent_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*layers)
        
        # Codebook
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, x):
        z = self.encoder(x)
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Quantize
        distances = torch.sum(z**2, dim=1, keepdim=True) + \
                   torch.sum(self.codebook.weight**2, dim=1) - \
                   2 * torch.matmul(z, self.codebook.weight.t())
        
        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices).view(b, h, w, c).permute(0, 3, 1, 2)
        
        return z_q, indices.view(b, h, w)


class GenerationDecoder(nn.Module):
    """VQVAE-style decoder for image generation"""
    def __init__(self, out_chans=3, embed_dim=256, n_layers=4, latent_dim=8):
        super().__init__()
        
        layers = [nn.Conv2d(latent_dim, embed_dim * (2 ** (n_layers-1)), 3, 1, 1)]
        
        for i in range(n_layers-1, 0, -1):
            in_ch = embed_dim * (2 ** i)
            out_ch = embed_dim * (2 ** (i-1))
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            ])
        
        layers.append(nn.Conv2d(embed_dim, out_chans, 3, 1, 1))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return torch.tanh(self.decoder(z))


class DualVisualEncoder(nn.Module):
    """Dual visual encoder that decouples understanding and generation"""
    def __init__(self, understanding_dim=768, generation_dim=256, codebook_size=16384):
        super().__init__()
        self.understanding_encoder = UnderstandingEncoder(embed_dim=understanding_dim)
        self.generation_encoder = GenerationEncoder(embed_dim=generation_dim, codebook_size=codebook_size)
        self.generation_decoder = GenerationDecoder(embed_dim=generation_dim)
    
    def encode_for_understanding(self, images):
        return self.understanding_encoder(images)
    
    def encode_for_generation(self, images):
        return self.generation_encoder(images)
    
    def decode_generation_tokens(self, z_q):
        return self.generation_decoder(z_q)
