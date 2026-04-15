#!/usr/bin/env python3
import sys
import types
import json
import torch
from collections import Counter
from torch_geometric.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
    return torch.load(path, weights_only=False)

def get_stats(dataset):
    stats = {
        'num_samples': len(dataset),
        'avg_num_nodes': 0.0,
        'avg_num_edges': 0.0,
        'feat_dim': dataset[0].x.shape[1],
        'edge_feat_dim': dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else None,
        'label_dist': {}
    }
    num_nodes_list = []
    num_edges_list = []
    ys = []
    for data in dataset:
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
    return stats

if __name__ == '__main__':
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
    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    datasets = [('Pretrain', stats['pretrain']), ('Finetune', stats['finetune']), ('Candidate', stats['candidate'])]
    for i, (name, s) in enumerate(datasets):
        axs[0,0].bar(range(len(s['label_dist'])), s['label_dist'].values(), tick_label=s['label_dist'].keys())
        axs[0,0].set_title(f'{name} Label Dist')
        axs[0,1].hist([s['avg_num_nodes']], bins=1) # dummy
        # Better: collect all
    # Num nodes hist
    all_nodes = []
    for name, s in datasets:
        # Need to collect
        pass  # simplify later
    plt.savefig('report/images/data_stats.png')
    plt.close()
    print('Saved data_stats.json and data_stats.png')
