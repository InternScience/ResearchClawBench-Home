import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from skimage import filters, feature, color

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'demo_imgs'
OUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid')


def load_rgb(path: Path):
    return np.array(Image.open(path).convert('RGB'))


def patch_metrics(img: np.ndarray, patch: int):
    h, w, _ = img.shape
    rows = []
    gray = color.rgb2gray(img)
    entropy_map = filters.rank.entropy((gray * 255).astype(np.uint8), np.ones((9, 9), dtype=np.uint8))
    edges = feature.canny(gray, sigma=1.2)
    for y in range(0, h, patch):
        for x in range(0, w, patch):
            patch_img = img[y:min(y+patch, h), x:min(x+patch, w)]
            patch_gray = gray[y:min(y+patch, h), x:min(x+patch, w)]
            patch_entropy = entropy_map[y:min(y+patch, h), x:min(x+patch, w)]
            patch_edges = edges[y:min(y+patch, h), x:min(x+patch, w)]
            rows.append({
                'x1': x, 'y1': y, 'x2': min(x+patch, w), 'y2': min(y+patch, h),
                'patch_size': patch,
                'area': patch_img.shape[0] * patch_img.shape[1],
                'mean_intensity': float(patch_gray.mean()),
                'std_intensity': float(patch_gray.std()),
                'entropy': float(patch_entropy.mean()),
                'edge_density': float(patch_edges.mean()),
                'saliency_score': float(0.45*patch_gray.std() + 0.35*patch_entropy.mean() + 0.20*(patch_edges.mean()*100)),
            })
    df = pd.DataFrame(rows)
    df['rank'] = df['saliency_score'].rank(ascending=False, method='dense').astype(int)
    return df


def draw_boxes(img: np.ndarray, boxes: pd.DataFrame, save_path: Path, color_name='red', width=6):
    im = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(im)
    for _, r in boxes.iterrows():
        draw.rectangle([int(r.x1), int(r.y1), int(r.x2), int(r.y2)], outline=color_name, width=width)
    im.save(save_path)


def crop_and_save(img: np.ndarray, row: pd.Series, save_path: Path):
    crop = Image.fromarray(img[int(row.y1):int(row.y2), int(row.x1):int(row.x2)])
    crop.save(save_path)


def main():
    image_paths = sorted(DATA_DIR.glob('*.png'))
    all_rows = []
    summary_rows = []

    for p in image_paths:
        img = load_rgb(p)
        h, w, _ = img.shape
        for patch in [128, 224, 336, 448]:
            df = patch_metrics(img, patch)
            df['image'] = p.name
            all_rows.append(df)
            top = df.sort_values('saliency_score', ascending=False).head(5).copy()
            top['relative_area'] = top['area'] / (h * w)
            summary_rows.append({
                'image': p.name,
                'patch_size': patch,
                'num_patches': len(df),
                'top1_score': float(top.iloc[0]['saliency_score']),
                'top5_mean_score': float(top['saliency_score'].mean()),
                'top5_mean_relative_area': float(top['relative_area'].mean()),
            })

        # produce qualitative overlays for 224 and 448 top-3 regions
        for patch in [224, 448]:
            df = patch_metrics(img, patch).sort_values('saliency_score', ascending=False).head(3)
            draw_boxes(img, df, IMG_DIR / f"{p.stem}_top_regions_{patch}.png")
            crop_and_save(img, df.iloc[0], IMG_DIR / f"{p.stem}_best_crop_{patch}.png")

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    all_df.to_csv(OUT_DIR / 'patch_metrics.csv', index=False)
    summary_df.to_csv(OUT_DIR / 'cropping_summary.csv', index=False)

    # Figure 1: data overview
    meta = []
    for p in image_paths:
        img = load_rgb(p)
        meta.append({'image': p.name, 'width': img.shape[1], 'height': img.shape[0], 'pixels_m': img.shape[0]*img.shape[1]/1e6})
    meta_df = pd.DataFrame(meta)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=meta_df, x='image', y='pixels_m', ax=axes[0], palette='viridis')
    axes[0].set_ylabel('Megapixels')
    axes[0].set_xlabel('Image')
    axes[0].set_title('Demo image scale')
    sns.scatterplot(data=meta_df, x='width', y='height', hue='image', s=120, ax=axes[1])
    axes[1].set_title('Resolution distribution')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'data_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: saliency score vs patch size
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=summary_df, x='patch_size', y='top5_mean_score', hue='image', marker='o', ax=ax)
    ax.set_title('Average saliency of top-5 patches across crop scales')
    ax.set_xlabel('Crop size (pixels)')
    ax.set_ylabel('Top-5 mean saliency score')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'scale_saliency_comparison.png', dpi=200)
    plt.close(fig)

    # Figure 3: relative area tradeoff
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=summary_df, x='patch_size', y='top5_mean_relative_area', hue='image', marker='o', ax=ax)
    ax.set_title('Area budget consumed by selected high-saliency crops')
    ax.set_xlabel('Crop size (pixels)')
    ax.set_ylabel('Mean relative area of top-5 patches')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'area_budget_tradeoff.png', dpi=200)
    plt.close(fig)

    # Figure 4: patch score distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    subset = all_df[all_df['patch_size'].isin([224, 448])].copy()
    sns.boxplot(data=subset, x='patch_size', y='saliency_score', hue='image', ax=ax)
    ax.set_title('Distribution of patch saliency scores')
    ax.set_xlabel('Patch size (pixels)')
    ax.set_ylabel('Saliency score')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'patch_score_distribution.png', dpi=200)
    plt.close(fig)

    # JSON summary
    best = all_df.sort_values('saliency_score', ascending=False).groupby('image').head(1)
    result = {
        'images': meta,
        'best_regions': best[['image', 'patch_size', 'x1', 'y1', 'x2', 'y2', 'saliency_score', 'entropy', 'edge_density']].to_dict(orient='records')
    }
    (OUT_DIR / 'analysis_summary.json').write_text(json.dumps(result, indent=2))
    print('Analysis complete')


if __name__ == '__main__':
    main()
