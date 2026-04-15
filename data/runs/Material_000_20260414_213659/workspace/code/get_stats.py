import sys
import types
import torch
import json
from collections import Counter
import numpy as np
from torch_geometric.data import Data

data_prepare_mod = types.ModuleType('data_prepare')
sys.modules['data_prepare'] = data_prepare_mod

class RealisticCrystalDataset:
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

def get_data_list(dataset):
    return dataset.data_list

def get_stats(dataset):
    data_list = get_data_list(dataset)
    stats = {
        'num_samples': len(dataset),
        'feat_dim': data_list[0].x.shape[1],
        'edge_feat_dim': data_list[0].edge_attr.shape[1] if data_list[0].edge_attr is not None else None,
        'avg_num_nodes': 0.0,
        'std_num_nodes': 0.0,
        'avg_num_edges': 0.0,
        'std_num_edges': 0.0,
        'label_dist': {}
    }
    num_nodes_list = []
    num_edges_list = []
    ys = []
    for data in data_list:
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)
        if hasattr(data, 'y') and data.y is not None:
            ys.append(data.y.item())
    stats['avg_num_nodes'] = np.mean(num_nodes_list)
    stats['std_num_nodes'] = np.std(num_nodes_list)
    stats['avg_num_edges'] = np.mean(num_edges_list)
    stats['std_num_edges'] = np.std(num_edges_list)
    stats['max_nodes'] = max(num_nodes_list)
    stats['max_edges'] = max(num_edges_list)
    if ys:
        label_count = Counter(ys)
        stats['label_dist'] = dict(label_count)
        stats['pos_ratio'] = label_count[1] / len(ys) if 1 in label_count else 0
    return stats

pre = load_dataset('data/pretrain_data.pt')
ft = load_dataset('data/finetune_data.pt')
cand = load_dataset('data/candidate_data.pt')
stats = {
    'pretrain': get_stats(pre),
    'finetune': get_stats(ft),
    'candidate': get_stats(cand)
}
print(stats)
with open('outputs/data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
print('Saved')
