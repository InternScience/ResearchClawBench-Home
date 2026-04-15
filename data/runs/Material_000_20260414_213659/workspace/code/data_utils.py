import sys
import types
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

data_prepare_mod = types.ModuleType('data_prepare')
sys.modules['data_prepare'] = data_prepare_mod

class RealisticCrystalDataset(Dataset):
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        data = self.data_list[idx]
        data.idx = torch.tensor([idx])
        return data

data_prepare_mod.RealisticCrystalDataset = RealisticCrystalDataset

def load_dataset(path):
    dataset = torch.load(path, weights_only=False)
    return dataset

def get_stats(dataset):
    stats = {
        'num_samples': len(dataset),
        'avg_num_nodes': 0.0,
        'avg_num_edges': 0.0,
        'feat_dim': dataset[0].x.shape[1] if hasattr(dataset[0], 'x') else None,
        'edge_feat_dim': dataset[0].edge_attr.shape[1] if hasattr(dataset[0], 'edge_attr') else None,
        'label_dist': {}
    }
    ys = []
    num_nodes_list = []
    num_edges_list = []
    for data in dataset.data_list:
        if hasattr(data, 'y'):
            ys.append(data.y.item())
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)
    if ys:
        stats['label_dist'] = {int(k): int(v) for k,v in torch.bincount(torch.tensor(ys)).items()}
    stats['avg_num_nodes'] = sum(num_nodes_list) / len(num_nodes_list)
    stats['avg_num_edges'] = sum(num_edges_list) / len(num_edges_list)
    stats['max_nodes'] = max(num_nodes_list)
    stats['max_edges'] = max(num_edges_list)
    return stats
