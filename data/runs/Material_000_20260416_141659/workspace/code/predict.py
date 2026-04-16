import sys
import types
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.modules['data_prepare'] = types.ModuleType('data_prepare')
class RealisticCrystalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
sys.modules['data_prepare'].RealisticCrystalDataset = RealisticCrystalDataset

class GNNEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = global_mean_pool(x, batch)
        return x

class ClassifierModel(nn.Module):
    def __init__(self, encoder, hidden_channels):
        super(ClassifierModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index, batch)
        out = self.classifier(h)
        return out

def predict_candidates():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load candidate data
    candidate_data = torch.load('data/candidate_data.pt', map_location=device, weights_only=False)
    loader = DataLoader(candidate_data, batch_size=64, shuffle=False)

    num_node_features = candidate_data[0].x.shape[1]
    hidden_channels = 128
    
    encoder = GNNEncoder(num_node_features, hidden_channels).to(device)
    model = ClassifierModel(encoder, hidden_channels).to(device)
    
    try:
        model.load_state_dict(torch.load('outputs/finetuned_model_best.pth', map_location=device))
        print("Loaded finetuned model weights.")
    except Exception as e:
        print(f"Could not load finetuned weights: {e}")
        return

    model.eval()
    all_preds = []
    all_labels = []  # Hidden true labels for evaluation
    
    print("Starting prediction...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch).view(-1)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            if hasattr(data, 'y') and data.y is not None:
                all_labels.extend(data.y.cpu().numpy())

    # Save predictions
    df = pd.DataFrame({'Candidate_ID': range(len(all_preds)), 'Altermagnet_Prob': all_preds})
    if len(all_labels) == len(all_preds):
        df['True_Label'] = all_labels
    
    df = df.sort_values(by='Altermagnet_Prob', ascending=False)
    df.to_csv('outputs/candidate_predictions.csv', index=False)
    print("Predictions saved to outputs/candidate_predictions.csv")

    # Plot distribution of probabilities
    plt.figure()
    plt.hist(all_preds, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Probability of being an Altermagnet')
    plt.ylabel('Frequency')
    plt.savefig('outputs/prob_distribution.png')
    
    if len(all_labels) == len(all_preds):
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        print(f"Candidate Evaluation - ROC AUC: {roc_auc:.4f}")
        
        pr_auc = average_precision_score(all_labels, all_preds)
        print(f"Candidate Evaluation - PR AUC: {pr_auc:.4f}")
        
        # Select Top 50 candidates and evaluate
        top_50 = df.head(50)
        true_positives_in_top_50 = top_50['True_Label'].sum()
        print(f"True Altermagnets in Top 50 Candidates: {true_positives_in_top_50}/50")
        
        # Top 100
        top_100 = df.head(100)
        true_positives_in_top_100 = top_100['True_Label'].sum()
        print(f"True Altermagnets in Top 100 Candidates: {true_positives_in_top_100}/100")

if __name__ == "__main__":
    predict_candidates()
