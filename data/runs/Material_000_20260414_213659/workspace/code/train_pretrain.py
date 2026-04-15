import torch
from torch_geometric.loader import DataLoader
from model import GINEncoder
from code.data_utils import load_dataset  # assume data_utils has load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset('data/pretrain_data.pt')
loader = DataLoader(dataset, batch_size=128, shuffle=True)

encoder = GINEncoder().to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

def contrastive_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(N).to(z1.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels) / 2

for epoch in range(50):
    encoder.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # Aug1: edge drop
        edge_index1, _ = dropout_edge(data.edge_index, p=0.1)
        h1 = encoder(data.x, edge_index1, data.batch)
        # Aug2
        edge_index2, _ = dropout_edge(data.edge_index, p=0.1)
        h2 = encoder(data.x, edge_index2, data.batch)
        loss = contrastive_loss(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(loader)}')

torch.save(encoder.state_dict(), 'outputs/pretrained_encoder.pt')
print('Pretrained saved')
