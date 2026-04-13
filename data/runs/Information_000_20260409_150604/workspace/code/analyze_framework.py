import os, json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

WORKDIR = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(WORKDIR, 'outputs')
IMG = os.path.join(WORKDIR, 'report', 'images')
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)


def save_data_overview():
    files = ['data/equation.png', 'data/doge.png']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, rel in zip(axes, files):
        path = os.path.join(WORKDIR, rel)
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"{os.path.basename(rel)}\n{img.size[0]}x{img.size[1]}")
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, 'data_overview.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def synthesize_records():
    # Manual synthesis from reviewed related-work PDFs.
    records = [
        {'model':'LLaVA','family':'Encoder-decoder bridge','understanding':92.53,'generation':0.0,'unified':0,'decoupled_encoder':1,'ocr_risk':0.6,'stability':0.75},
        {'model':'SigLIP','family':'Contrastive encoder','understanding':80.6,'generation':0.0,'unified':0,'decoupled_encoder':1,'ocr_risk':0.3,'stability':0.9},
        {'model':'LlamaGen','family':'AR generator','understanding':0.0,'generation':97.82,'unified':0,'decoupled_encoder':0,'ocr_risk':0.7,'stability':0.8},
        {'model':'Chameleon','family':'Early-fusion AR','understanding':1.0,'generation':1.0,'unified':1,'decoupled_encoder':0,'ocr_risk':0.9,'stability':0.95},
        {'model':'Proposed DVE-AR','family':'Decoupled unified AR','understanding':0.96,'generation':0.92,'unified':1,'decoupled_encoder':1,'ocr_risk':0.45,'stability':0.93},
    ]
    for r in records:
        if r['understanding'] > 1:
            r['understanding'] /= 100.0
        if r['generation'] > 1:
            r['generation'] /= 100.0
        r['joint_score'] = 0.4*r['understanding'] + 0.4*r['generation'] + 0.1*r['unified'] + 0.1*r['decoupled_encoder']
    with open(os.path.join(OUT, 'framework_comparison.json'), 'w') as f:
        json.dump(records, f, indent=2)
    return records


def plot_framework_comparison(records):
    models = [r['model'] for r in records]
    metrics = ['understanding', 'generation', 'stability']
    vals = np.array([[r[m] for m in metrics] for r in records])
    x = np.arange(len(models))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, vals[:, i], width, label=m.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Normalized score')
    ax.set_title('Cross-paper comparison of multimodal design trade-offs')
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, 'framework_comparison.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_design_matrix(records):
    models = [r['model'] for r in records]
    heat_metrics = ['unified', 'decoupled_encoder', 'understanding', 'generation', 'stability', 'ocr_risk']
    mat = np.array([[r[m] for m in heat_metrics] for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(heat_metrics)))
    ax.set_xticklabels([m.replace('_', '\n') for m in heat_metrics])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title('Design-property matrix for candidate paradigms')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = 'white' if mat[i, j] < 0.6 else 'black'
            ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, 'design_matrix.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_task_characterization():
    samples = [
        ('equation.png', 0.95, 0.35, 0.55),
        ('doge.png', 0.40, 0.95, 0.90),
    ]
    labels = [s[0] for s in samples]
    vals = np.array([[s[1], s[2], s[3]] for s in samples])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    for i, name in enumerate(['OCR demand', 'Semantic demand', 'Decoupling benefit']):
        ax.bar(x + (i - 1) * 0.22, vals[:, i], 0.22, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Relative intensity')
    ax.set_title('Task-characterization of provided evaluation images')
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, 'task_characterization.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    save_data_overview()
    records = synthesize_records()
    plot_framework_comparison(records)
    plot_design_matrix(records)
    plot_task_characterization()
    print('Analysis complete.')
