import sys
import types
import torch
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
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

def train_finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load finetune data
    finetune_data = torch.load('data/finetune_data.pt', map_location=device, weights_only=False)
    
    # Split into train and validation
    train_size = int(0.8 * len(finetune_data))
    val_size = len(finetune_data) - train_size
    train_dataset, val_dataset = random_split(finetune_data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_node_features = finetune_data[0].x.shape[1]
    hidden_channels = 128
    
    encoder = GNNEncoder(num_node_features, hidden_channels).to(device)
    try:
        encoder.load_state_dict(torch.load('outputs/pretrained_encoder.pth', map_location=device))
        print("Loaded pretrained encoder weights.")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")

    model = ClassifierModel(encoder, hidden_channels).to(device)
    
    # Calculate pos_weight for BCEWithLogitsLoss due to class imbalance
    y_vals = [data.y.item() for data in train_dataset]
    num_pos = sum(y_vals)
    num_neg = len(y_vals) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).to(device)
    print(f"Class imbalance - Pos: {num_pos}, Neg: {num_neg}, pos_weight: {pos_weight.item():.2f}")

    # Focal loss alternative or weighted BCE
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    print("Starting finetuning...")
    epochs = 60
    train_losses = []
    val_losses = []
    val_aucs = []

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch).view(-1)
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch).view(-1)
                loss = criterion(out, data.y.float())
                val_loss += loss.item()
                
                preds = torch.sigmoid(out).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(data.y.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except:
            val_auc = 0.0
        val_aucs.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'outputs/finetuned_model_best.pth')

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Finetuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/finetune_loss.png')

    # Load best model for final evaluation
    model.load_state_dict(torch.load('outputs/finetuned_model_best.pth', map_location=device))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch).view(-1)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Validation)')
    plt.legend(loc="lower right")
    plt.savefig('report/images/roc_curve.png')

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Validation)')
    plt.legend(loc="lower left")
    plt.savefig('report/images/pr_curve.png')

    print(f"Finetuning complete. Best Val AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    train_finetune()
