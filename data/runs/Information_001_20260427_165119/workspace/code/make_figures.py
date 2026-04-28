"""Generate all figures for the report."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import zoom as ndzoom

sys.path.insert(0, '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/code')

WS  = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119'
IMG = os.path.join(WS, 'data/demo_imgs')
OUT = os.path.join(WS, 'outputs')
FIG = os.path.join(WS, 'report/images')
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(OUT, 'per_demo_predictions.json')) as f:
    results = json.load(f)
with open(os.path.join(OUT, 'main_results.json')) as f:
    main = json.load(f)
with open(os.path.join(OUT, 'ablations.json')) as f:
    abl = json.load(f)


def heatmap_overlay(img: np.ndarray, m: np.ndarray, alpha=0.5) -> np.ndarray:
    H, W = img.shape[:2]
    g = m.shape[0]
    up = ndzoom(m, (H/g, W/g), order=1)
    if up.max() > up.min():
        up = (up - up.min()) / (up.max() - up.min() + 1e-9)
    cmap = plt.get_cmap('jet')(up)[..., :3]
    return np.clip((1 - alpha) * (img/255.) + alpha * cmap, 0, 1)


# ============================================================
# Figure 1: Data overview
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, fn, title in zip(
    axes,
    ['demo1.png', 'demo2.png', 'method_case.png'],
    ['demo1.png — Street scene\n(1024×768)',
     'demo2.png — Greenhouse tulips\n(2250×1500)',
     'method_case.png — Paper teaser\n(2500×1681, qualitative)'],
):
    pil = Image.open(os.path.join(IMG, fn)).convert('RGB')
    ax.imshow(pil)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
plt.suptitle('Data overview — demo images for ViCrop reproduction', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'data_overview.png'), dpi=130, bbox_inches='tight')
plt.close()
print('Saved data_overview.png')


# ============================================================
# Figure 2: Relevancy heatmaps + bbox overlays for each task
# ============================================================
n = len(results)
fig, axes = plt.subplots(n, 3, figsize=(15, 3.6*n))
if n == 1:
    axes = axes[None, :]
for i, r in enumerate(results):
    pil = Image.open(os.path.join(IMG, r['image'])).convert('RGB')
    arr = np.array(pil)
    rel = np.load(os.path.join(OUT, f'rel_chefer_T{i}.npy'))
    roll = np.load(os.path.join(OUT, f'rel_rollout_T{i}.npy'))
    bbox = r['bbox']
    # (a) original + bbox
    axes[i, 0].imshow(arr)
    rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                               linewidth=3, edgecolor='cyan', facecolor='none')
    axes[i, 0].add_patch(rect)
    axes[i, 0].set_title(f'T{i}: {r["short_question"]}\noriginal + ViCrop bbox',
                         fontsize=10)
    axes[i, 0].axis('off')
    # (b) Chefer relevancy overlay
    axes[i, 1].imshow(heatmap_overlay(arr, rel, alpha=0.55))
    axes[i, 1].set_title('Chefer attention×grad relevancy', fontsize=10)
    axes[i, 1].axis('off')
    # (c) zoomed crop
    crop = pil.crop(bbox)
    axes[i, 2].imshow(np.array(crop))
    pred_g = r['short_options'][r['pred_global_idx']]
    pred_c = r['short_options'][r['pred_crop_idx']]
    pred_e = r['short_options'][r['pred_vicrop_idx']]
    gold = r['gold_option']
    color = 'green' if r['pred_vicrop_idx'] == r['gold_idx'] else 'red'
    axes[i, 2].set_title(
        f'cropped view\nGT={gold}  global→{pred_g}  crop→{pred_c}\nViCrop→{pred_e}',
        fontsize=10, color=color)
    axes[i, 2].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'relevancy_heatmaps.png'), dpi=120, bbox_inches='tight')
plt.close()
print('Saved relevancy_heatmaps.png')


# ============================================================
# Figure 3: Crop overlay grid (compact bbox + crop view)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
flat = axes.flatten()
for i, r in enumerate(results):
    if i >= len(flat): break
    pil = Image.open(os.path.join(IMG, r['image'])).convert('RGB')
    arr = np.array(pil)
    bbox = r['bbox']
    rel = np.load(os.path.join(OUT, f'rel_chefer_T{i}.npy'))
    over = heatmap_overlay(arr, rel, alpha=0.4)
    flat[i].imshow(over)
    rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                               linewidth=3, edgecolor='cyan', facecolor='none')
    flat[i].add_patch(rect)
    correct = r['pred_vicrop_idx'] == r['gold_idx']
    flat[i].set_title(f'T{i}: {r["short_question"]}\n'
                      f'GT={r["gold_option"]} · ViCrop→{r["short_options"][r["pred_vicrop_idx"]]} '
                      f'{"✓" if correct else "✗"}',
                      fontsize=10, color='green' if correct else 'red')
    flat[i].axis('off')
for j in range(len(results), len(flat)):
    flat[j].axis('off')
plt.suptitle('ViCrop ROI localisation across all six tasks', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'crop_overlays.png'), dpi=120, bbox_inches='tight')
plt.close()
print('Saved crop_overlays.png')


# ============================================================
# Figure 4: Baseline accuracy comparison
# ============================================================
acc = main['accuracy']
order = ['no_crop', 'center_crop', 'random_crop', 'uniform_tile',
         'vicrop_global_only', 'vicrop_crop_only', 'vicrop_ensemble']
labels = ['no-crop\n(vanilla CLIP)', 'center-crop\n(50%)', 'random-crop\n(50%)',
          'uniform 2×2 tile\n(Monkey-style)',
          'ViCrop\nglobal only', 'ViCrop\ncrop only', 'ViCrop\nensemble']
colors = ['#888888', '#aaaaaa', '#bbbbbb', '#bbcccc',
          '#3380ff', '#ff8033', '#33aa33']
vals = [acc[k] for k in order]
fig, ax = plt.subplots(figsize=(11, 4.2))
bars = ax.bar(range(len(order)), vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2%}',
            ha='center', fontsize=10)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel('Accuracy on 6 fine-grained MC tasks')
ax.set_title('Baseline comparison — task-guided ViCrop crops vs. naïve crops')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'baseline_comparison.png'), dpi=130, bbox_inches='tight')
plt.close()
print('Saved baseline_comparison.png')


# ============================================================
# Figure 5: Ablations — threshold and ensemble weight
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# threshold
axes[0].plot(abl['threshold']['thresholds'], abl['threshold']['accuracy'],
             marker='o', color='#33aa33')
axes[0].set_xlabel('Relevancy threshold quantile')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Effect of relevancy threshold on ViCrop')
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(abl['w_crop']['w_crop'], abl['w_crop']['accuracy'],
         marker='o', color='#3380ff', label='accuracy')
ax2.set_xlabel('Crop weight w (1−w on global, w on crop)')
ax2.set_ylabel('Accuracy', color='#3380ff')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelcolor='#3380ff')
ax3 = ax2.twinx()
ax3.plot(abl['w_crop']['w_crop'], abl['w_crop']['mean_p_correct'],
         marker='s', linestyle='--', color='#ff8033', label='P(correct)')
ax3.set_ylabel('mean P(correct option)', color='#ff8033')
ax3.tick_params(axis='y', labelcolor='#ff8033')
ax2.set_title('Effect of crop/global ensemble weight')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'ablation_threshold.png'), dpi=130, bbox_inches='tight')
plt.close()
print('Saved ablation_threshold.png')


# ============================================================
# Figure 6: Per-task probability bars
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
flat = axes.flatten()
for i, r in enumerate(results):
    if i >= len(flat): break
    ax = flat[i]
    K = len(r['short_options'])
    x = np.arange(K)
    w = 0.27
    ax.bar(x - w, r['p_global'], w, label='global', color='#3380ff')
    ax.bar(x,     r['p_crop'],   w, label='crop',   color='#ff8033')
    ax.bar(x + w, r['p_ens'],    w, label='ensemble', color='#33aa33')
    ax.set_xticks(x)
    ax.set_xticklabels(r['short_options'], rotation=20, fontsize=9)
    correct = r['pred_vicrop_idx'] == r['gold_idx']
    ax.set_title(f'T{i}: {r["short_question"]}\nGT={r["gold_option"]} {"✓" if correct else "✗"}',
                 fontsize=10, color='green' if correct else 'red')
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5,
               label='chance (1/4)' if i == 0 else None)
    if i == 0:
        ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('softmax prob.')
for j in range(len(results), len(flat)):
    flat[j].axis('off')
plt.suptitle('Option probabilities under global / crop / ensemble views', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'per_task_probabilities.png'), dpi=120, bbox_inches='tight')
plt.close()
print('Saved per_task_probabilities.png')

print('\nAll figures saved to', FIG)
