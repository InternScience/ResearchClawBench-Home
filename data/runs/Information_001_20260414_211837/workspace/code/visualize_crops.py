import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import shutil

data_dir = Path('data/demo_imgs')
report_img_dir = Path('report/images')
report_img_dir.mkdir(parents=True, exist_ok=True)

images = {
    'demo1.png': {
        'description': 'Urban street with taxis and license plates as small objects.',
        'rois': [
            {'name': 'Silver car plate TA1J90', 'bbox': [[720, 440], [860, 440], [860, 470], [720, 470]]},
            {'name': 'Left taxi plate', 'bbox': [[25, 525], [125, 525], [125, 555], [25, 555]]},
            {'name': 'Right taxi plate', 'bbox': [[950, 500], [1050, 500], [1050, 530], [950, 530]]},
        ]
    },
    'demo2.png': {
        'description': 'Flower market with colorful tulips; fine-grained color/texture details.',
        'rois': [
            {'name': 'Single tulip closeup', 'bbox': [[1000, 800], [1200, 800], [1200, 1000], [1000, 1000]]},
            {'name': 'Yellow flower cluster', 'bbox': [[1800, 900], [2100, 900], [2100, 1100], [1800, 1100]]},
        ]
    },
    'method_case.png': {
        'description': 'Paper figure showing MLLM failures on small details (clock, list, player name).',
        'rois': [
            {'name': 'Clock inset', 'bbox': [[200, 50], [400, 50], [400, 250], [200, 250]]},
            {'name': 'Bookstore shelf', 'bbox': [[400, 300], [900, 300], [900, 800], [400, 800]]},
            {'name': 'Player jersey', 'bbox': [[1400, 900], [1900, 900], [1900, 1300], [1400, 1300]]},
        ]
    }
}

# Copy originals
for img_name in images:
    shutil.copy2(data_dir / img_name, report_img_dir / img_name)

fig_count = 0

# Data overview montage
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for i, img_name in enumerate(images):
    img = plt.imread(str(data_dir / img_name))
    axs[i].imshow(img)
    axs[i].set_title(images[img_name]['description'], fontsize=12)
    axs[i].axis('off')
plt.tight_layout()
plt.savefig(report_img_dir / 'data_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# Per image ROI visualization
for img_name, info in images.items():
    img = cv2.imread(str(data_dir / img_name))
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Overlay ROIs
    overlay = img_rgb.copy()
    for roi in info['rois']:
        bbox = np.array(roi['bbox'], dtype=np.int32)
        cv2.polylines(overlay, [bbox], True, (0, 255, 0), 3)
        # Label
        x,y = np.min(bbox[:,0]), np.min(bbox[:,1])
        cv2.putText(overlay, roi['name'][:20], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    plt.figure(figsize=(15,10))
    plt.imshow(overlay)
    plt.title(f'Proposed Task-Guided ROIs for Fine-Grained Details in {img_name}')
    plt.axis('off')
    plt.savefig(report_img_dir / f'{img_name[:-4]}_rois_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Crops
    for j, roi in enumerate(info['rois']):
        bbox = np.array(roi['bbox'], dtype=np.int32)
        x,y,w_r,h_r = cv2.boundingRect(bbox)
        # Clamp
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        w_r = min(w_r, w - x)
        h_r = min(h_r, h - y)
        crop = img[y:y+h_r, x:x+w_r]
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (224, 224))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6,6))
            plt.imshow(crop_rgb)
            plt.title(f'Zoomed {roi[\"name\"]} (224x224)')
            plt.axis('off')
            plt.savefig(report_img_dir / f'{img_name[:-4]}_crop_{j}.png', dpi=150, bbox_inches='tight')
            plt.close()

# Summary table
summary = []
for img_name, info in images.items():
    h,w,_ = plt.imread(str(data_dir / img_name)).shape[:2]
    summary.append({'Image': img_name, 'Resolution': f'{w}x{h}', 'Num ROIs': len(info['rois'])})

fig, ax = plt.subplots(figsize=(8,3))
ax.axis('off')
table = ax.table(cellText=[[r['Image'], r['Resolution'], r['Num ROIs']] for r in summary],
                 colLabels=['Image', 'Resolution', 'Num ROIs'],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
plt.savefig(report_img_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
plt.close()

print('Figures generated!')
