import matplotlib.pyplot as plt
import numpy as np

datasets = ["BACE", "BBBP", "ClinTox", "HIV", "MUV"]
ka_aucs = []
base_aucs = []
ka_times = []
base_times = []

for ds in datasets:
    with open(f"outputs/results_{ds.lower()}.txt") as f:
        lines = f.readlines()
        
        ka_auc = float(lines[2].split(":")[1].strip())
        ka_time = float(lines[3].split(":")[1].replace("s", "").strip())
        
        base_auc = float(lines[5].split(":")[1].strip())
        base_time = float(lines[6].split(":")[1].replace("s", "").strip())
        
        ka_aucs.append(ka_auc)
        base_aucs.append(base_auc)
        ka_times.append(ka_time)
        base_times.append(base_time)

x = np.arange(len(datasets))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, ka_aucs, width, label='KA-GNN', color='skyblue')
plt.bar(x + width/2, base_aucs, width, label='Baseline GNN', color='salmon')
plt.xlabel('Dataset')
plt.ylabel('Test ROC-AUC')
plt.title('Test ROC-AUC Comparison')
plt.xticks(x, datasets)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("report/images/summary_auc.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, ka_times, width, label='KA-GNN', color='skyblue')
plt.bar(x + width/2, base_times, width, label='Baseline GNN', color='salmon')
plt.xlabel('Dataset')
plt.ylabel('Avg Epoch Train Time (s)')
plt.title('Training Time Comparison')
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("report/images/summary_time.png")
plt.close()
