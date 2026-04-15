import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data
with open('outputs/dataset.json', 'r') as f:
    data = json.load(f)

a_params = np.array(data['structure_generation'][0])
b_params = np.array(data['structure_generation'][1])

data_tensor = torch.tensor(np.column_stack([a_params, b_params]), dtype=torch.float32)

# Simple VAE
class VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_vae(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl_loss

# Train
dataset = TensorDataset(data_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    train_loss = 0
    for batch in loader:
        x = batch[0]
        recon, mu, logvar = model(x)
        loss = loss_vae(recon, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {train_loss/len(loader):.4f}')

torch.save(model.state_dict(), 'outputs/models/structure_vae.pth')

# Generate samples
model.eval()
with torch.no_grad():
    z = torch.randn(100, 4)
    gen = model.decode(z).numpy()

# Plot
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(a_params, b_params, alpha=0.6, label='Real')
ax[0].set_xlabel('a')
ax[0].set_ylabel('b')
ax[0].set_title('Real Lattice Params')
ax[0].legend()

ax[1].scatter(gen[:,0], gen[:,1], alpha=0.6, color='orange', label='Generated')
ax[1].set_xlabel('a')
ax[1].set_ylabel('b')
ax[1].set_title('Generated Lattice Params')
ax[1].legend()

plt.tight_layout()
plt.savefig('report/images/structure_generation.png', dpi=300, bbox_inches='tight')
plt.close()

# Save
np.savez('outputs/structure_samples.npz', real=np.column_stack([a_params, b_params]), gen=gen)
