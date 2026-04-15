import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'demo_imgs'
OUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'report' / 'images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)


def grayscale(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def coarse_view(gray, target_min=336):
    h, w = gray.shape
    factor = max(1, int(round(min(h, w) / target_min)))
    small = gray[::factor, ::factor]
    up = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)[:h, :w]
    return up, factor


def edge_map(gray):
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return gx + gy


def roi_from_saliency(gray):
    h, w = gray.shape
    grad = edge_map(gray)
    cell = 32 if min(h, w) > 900 else 16
    hs, ws = h // cell, w // cell
    pooled = grad[:hs * cell, :ws * cell].reshape(hs, cell, ws, cell).mean((1, 3))
    y, x = np.unravel_index(np.argmax(pooled), pooled.shape)
    bw = bh = 4 * cell
    x0, y0 = x * cell, y * cell
    x1, y1 = min(w, x0 + bw), min(h, y0 + bh)
    return [int(x0), int(y0), int(x1), int(y1)], pooled


def edge_density(arr, thresh=20):
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    g = gx[:-1, :] + gy[:, :-1]
    return float((g > thresh).mean())


def contrast(arr):
    return float(arr.std())


def textlikeness(gray):
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return float(((gx > 18) & (gy < 12)).mean())


def annotate_roi(image_path, box, out_path):
    im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    draw.rectangle(box, outline=(0, 255, 255), width=6)
    im.save(out_path)


def heatmap_overlay(gray, pooled, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(gray, cmap='gray')
    pooled_up = np.kron(pooled, np.ones((gray.shape[0] // pooled.shape[0], gray.shape[1] // pooled.shape[1])))
    pooled_up = pooled_up[:gray.shape[0], :gray.shape[1]]
    ax.imshow(pooled_up, cmap='magma', alpha=0.45)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_image(path):
    img = load_rgb(path)
    gray = grayscale(img)
    global_view, factor = coarse_view(gray)
    box, pooled = roi_from_saliency(gray)
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    row = {
        'file': path.name,
        'width': img.shape[1],
        'height': img.shape[0],
        'downsample_factor': factor,
        'roi_x0': x0,
        'roi_y0': y0,
        'roi_x1': x1,
        'roi_y1': y1,
        'global_contrast': contrast(global_view),
        'roi_contrast': contrast(crop),
        'global_edge_density': edge_density(global_view),
        'roi_edge_density': edge_density(crop),
        'global_textlike_density': textlikeness(global_view),
        'roi_textlike_density': textlikeness(crop),
    }
    row['contrast_gain'] = row['roi_contrast'] / (row['global_contrast'] + 1e-6)
    row['edge_gain'] = row['roi_edge_density'] / (row['global_edge_density'] + 1e-6)
    row['textlike_gain'] = row['roi_textlike_density'] / (row['global_textlike_density'] + 1e-6)
    return gray, pooled, row, box


def make_qualitative_figure(rows):
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for i, row in enumerate(rows):
        path = DATA_DIR / row['file']
        rgb = np.asarray(Image.open(path).convert('RGB'))
        gray = grayscale(rgb.astype(np.float32))
        box = [row['roi_x0'], row['roi_y0'], row['roi_x1'], row['roi_y1']]
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        coarse, _ = coarse_view(gray)
        axes[i, 0].imshow(rgb)
        rect = plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec='cyan', lw=3)
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f"{row['file']} with proposed ROI")
        axes[i, 1].imshow(coarse, cmap='gray')
        axes[i, 1].set_title('Coarse fixed-resolution proxy')
        axes[i, 2].imshow(crop)
        axes[i, 2].set_title('Local crop / zoom-in detail')
        for j in range(3):
            axes[i, j].axis('off')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'qualitative_comparison.png', dpi=180)
    plt.close(fig)


def make_metric_figure(df):
    metrics = ['contrast_gain', 'edge_gain', 'textlike_gain']
    labels = ['Contrast gain', 'Edge density gain', 'Text-like density gain']
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + (idx-1)*width, df[metric].values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(df['file'].tolist())
    ax.set_ylabel('ROI / coarse-view ratio')
    ax.set_title('Information recovery proxies from task-guided cropping')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'metric_comparison.png', dpi=180)
    plt.close(fig)


def make_summary_json(df):
    summary = {
        'mean_contrast_gain': float(df['contrast_gain'].mean()),
        'mean_edge_gain': float(df['edge_gain'].mean()),
        'mean_textlike_gain': float(df['textlike_gain'].mean()),
        'files': df[['file', 'roi_x0', 'roi_y0', 'roi_x1', 'roi_y1']].to_dict(orient='records')
    }
    (OUT_DIR / 'summary_metrics.json').write_text(json.dumps(summary, indent=2))


def main():
    rows = []
    roi_boxes = {}
    for path in sorted(DATA_DIR.glob('demo*.png')):
        gray, pooled, row, box = analyze_image(path)
        rows.append(row)
        roi_boxes[path.name] = {'roi_box': box}
        annotate_roi(path, box, FIG_DIR / f'{path.stem}_roi.png')
        heatmap_overlay(gray, pooled, FIG_DIR / f'{path.stem}_heatmap.png')
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'image_metrics.csv', index=False)
    (OUT_DIR / 'roi_boxes.json').write_text(json.dumps(roi_boxes, indent=2))
    make_summary_json(df)
    make_qualitative_figure(rows)
    make_metric_figure(df)
    claims = [
        {
            'claim': 'Task-guided cropping increases local edge density relative to a coarse fixed-resolution proxy.',
            'artifact': 'outputs/image_metrics.csv',
            'support': df[['file', 'edge_gain']].to_dict(orient='records')
        },
        {
            'claim': 'The workspace qualitative method_case example visually shows improved fine-grained answers after crop-based zoom.',
            'artifact': 'data/demo_imgs/method_case.png',
            'support': 'ReadImage evidence in runtime plus report discussion.'
        },
        {
            'claim': 'Interpretability-style heatmaps can identify candidate ROIs aligned with high-frequency visual content.',
            'artifact': 'report/images/demo1_heatmap.png and report/images/demo2_heatmap.png',
            'support': 'Generated saliency overlays.'
        }
    ]
    (OUT_DIR / 'claim_recovery_table.json').write_text(json.dumps(claims, indent=2))

if __name__ == '__main__':
    main()
