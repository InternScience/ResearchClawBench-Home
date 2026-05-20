"""
Tiny VQ-VAE for 32x32 RGB images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)  # (B*H*W,)
        quantized = self.embeddings(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        # straight-through estimator
        z_q = z + (quantized - z).detach()
        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return z_q, loss, indices.view(B, H, W)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels//2, 4, stride=2, padding=1),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, hidden_channels, 4, stride=2, padding=1),  # 16->8
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_dim, 3, stride=1, padding=1),  # 8->8
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=64, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, 3, stride=1, padding=1),  # 8->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels//2, out_channels, 4, stride=2, padding=1),  # 16->32
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, latent_dim=64, num_embeddings=256, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels)

    def encode(self, x):
        z = self.encoder(x)
        z_q, loss, indices = self.quantizer(z)
        return z_q, loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        return x_recon, loss, recon_loss, vq_loss, indices


if __name__ == '__main__':
    model = VQVAE()
    x = torch.randn(2, 3, 32, 32)
    recon, loss, rloss, vqloss, idx = model(x)
    print("Recon shape:", recon.shape, "Loss:", loss.item())
