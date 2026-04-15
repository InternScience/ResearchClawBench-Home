import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SAGEConv
from torch_geometric.temporal.conv import TGNMemory
from sklearn.metrics import f1_score, accuracy_score
import json
import numpy as np

# Subsample data
data = torch.load('../data/NF-UNSW-NB15-v2_3d.pt', weights_only=False)
train_idx = torch.load('../outputs/train_idx.pt')[::10]
test_idx = torch.load('../outputs/test_idx.pt')[::5]
train_data = data[train_idx]
test_data = data[test_idx]

class StatDisentangle(nn.Module):
    def __init__(self, feat_dim=40):
        super().__init__()
        self.w = nn.Parameter(torch.ones(feat_dim)/feat_dim)
    
    def forward(self, x):
        return torch.sum(self.w * x, dim=1)  # simple weighted, train to decorrelate

class RepEncoder(nn.Module):
    def __init__(self, in_dim=40, hid=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DIDSSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.disent = StatDisentangle()
        self.enc = RepEncoder()
        self.classifier_bin = nn.Linear(64, 1)
        self.classifier_multi = nn.Linear(64, 10)
    
    def forward(self, x):
        xd = self.disent(x)
        # Simple rep disent: L1 on reps or corr low
        h = self.enc(xd.unsqueeze(1).repeat(1, x.size(1),1).view(-1,40) )  # dummy multi scale
        h = h.view(-1, x.size(1), 64).mean(1)
        bin_logit = self.classifier_bin(h).squeeze()
        multi_logit = self.classifier_multi(h)
        return bin_logit, multi_logit

model = DIDSSimple()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train = train_data.msg
y_train_bin = train_data.label.float()
y_train_multi = train_data.attack

for epoch in range(50):
    optimizer.zero_grad()
    bin_log, multi_log = model(X_train)
    loss_bin = F.binary_cross_entropy_with_logits(bin_log, y_train_bin)
    loss_multi = F.cross_entropy(multi_log, y_train_multi)
    loss = loss_bin + loss_multi + 0.1 * torch.mean(torch.abs(model.enc.fc2.weight))  # simple rep disent reg
    loss.backward()
    optimizer.step()

with torch.no_grad():
    bin_log, multi_log = model(test_data.msg)
    bin_pred = (torch.sigmoid(bin_log) > 0.5).float()
    multi_pred = multi_log.argmax(1)
    bin_acc = accuracy_score(test_data.label.numpy(), bin_pred.numpy())
    bin_f1 = f1_score(test_data.label.numpy(), bin_pred.numpy(), average='macro')
    multi_f1 = f1_score(test_data.attack.numpy(), multi_pred.numpy(), average='macro')
    
dids_results = {'bin_acc': float(bin_acc), 'bin_f1': float(bin_f1), 'multi_f1': float(multi_f1)}
print(json.dumps(dids_results))
torch.save(dids_results, '../outputs/dids_simple_results.json')
