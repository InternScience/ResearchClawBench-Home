import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from models import KAGNN, BaselineGNN
import matplotlib.pyplot as plt
import os
import gc

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # Handle missing labels (NaNs)
        is_valid = ~torch.isnan(data.y)
        loss = criterion(out[is_valid], data.y[is_valid])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(torch.sigmoid(out).cpu().numpy())
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Calculate ROC-AUC for each task, ignoring NaNs
    aucs = []
    for i in range(y_true.shape[1]):
        valid_idx = ~np.isnan(y_true[:, i])
        if np.sum(valid_idx) > 0 and len(np.unique(y_true[valid_idx, i])) > 1:
            auc = roc_auc_score(y_true[valid_idx, i], y_pred[valid_idx, i])
            aucs.append(auc)
    
    if len(aucs) == 0:
        return 0.5
    return np.mean(aucs)

def run_experiment(dataset_name, data_path, num_classes, epochs=30):
    print(f"Running experiment for {dataset_name}...")
    dataset = torch.load(data_path, weights_only=False)
    
    # Split dataset
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    train_dataset = [dataset[i] for i in indices[:train_size]]
    val_dataset = [dataset[i] for i in indices[train_size:train_size+val_size]]
    test_dataset = [dataset[i] for i in indices[train_size+val_size:]]
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_features = dataset[0].x.shape[1]
    hidden_dim = 64
    
    # Initialize models
    ka_model = KAGNN(node_features, hidden_dim, num_classes).to(device)
    base_model = BaselineGNN(node_features, hidden_dim, num_classes).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    results = {}
    
    for model_name, model in [("KA-GNN", ka_model), ("Baseline", base_model)]:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_auc = 0
        test_auc_at_best_val = 0
        
        train_times = []
        val_aucs = []
        
        for epoch in range(epochs):
            start_time = time.time()
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            train_time = time.time() - start_time
            train_times.append(train_time)
            
            val_auc = eval_epoch(model, val_loader, device)
            val_aucs.append(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_auc_at_best_val = eval_epoch(model, test_loader, device)
                
            if (epoch + 1) % 10 == 0:
                print(f"[{model_name}] Epoch {epoch+1:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
                
        results[model_name] = {
            "test_auc": test_auc_at_best_val,
            "avg_train_time": np.mean(train_times),
            "val_aucs": val_aucs
        }
        
    # Plot learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(results["KA-GNN"]["val_aucs"], label="KA-GNN")
    plt.plot(results["Baseline"]["val_aucs"], label="Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Validation ROC-AUC")
    plt.title(f"Validation ROC-AUC over Epochs ({dataset_name})")
    plt.legend()
    plt.savefig(f"report/images/val_curve_{dataset_name.lower()}.png")
    plt.close()
    
    # Save results
    with open(f"outputs/results_{dataset_name.lower()}.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        for model_name, res in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Test ROC-AUC: {res['test_auc']:.4f}\n")
            f.write(f"  Avg Epoch Time: {res['avg_train_time']:.4f} s\n")
            
    return results

if __name__ == "__main__":
    datasets = [
        ("BACE", "outputs/bace.pt", 1),
        ("BBBP", "outputs/bbbp.pt", 1),
        ("ClinTox", "outputs/clintox.pt", 2),
        ("HIV", "outputs/hiv.pt", 1),
        ("MUV", "outputs/muv.pt", 17)
    ]
    
    all_results = {}
    for name, path, num_classes in datasets:
        if os.path.exists(path):
            all_results[name] = run_experiment(name, path, num_classes, epochs=30)
            gc.collect()
            
    # Summary plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(all_results))
    width = 0.35
    
    ka_aucs = [all_results[name]["KA-GNN"]["test_auc"] for name in all_results]
    base_aucs = [all_results[name]["Baseline"]["test_auc"] for name in all_results]
    
    plt.bar(x - width/2, ka_aucs, width, label='KA-GNN')
    plt.bar(x + width/2, base_aucs, width, label='Baseline')
    
    plt.xlabel('Dataset')
    plt.ylabel('Test ROC-AUC')
    plt.title('Test ROC-AUC Comparison')
    plt.xticks(x, list(all_results.keys()))
    plt.legend()
    plt.savefig("report/images/summary_auc.png")
    plt.close()
    
    # Time comparison plot
    plt.figure(figsize=(10, 6))
    ka_times = [all_results[name]["KA-GNN"]["avg_train_time"] for name in all_results]
    base_times = [all_results[name]["Baseline"]["avg_train_time"] for name in all_results]
    
    plt.bar(x - width/2, ka_times, width, label='KA-GNN')
    plt.bar(x + width/2, base_times, width, label='Baseline')
    
    plt.xlabel('Dataset')
    plt.ylabel('Avg Epoch Train Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(x, list(all_results.keys()))
    plt.legend()
    plt.savefig("report/images/summary_time.png")
    plt.close()
