import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import json
from PIL import Image
import shutil

# Setup
data_dir = Path('data/demo_imgs')
output_dir = Path('outputs')
report_img_dir = Path('report/images')
report_img_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

images = ['demo1.png', 'demo2.png', 'method_case.png']
image_paths = [data_dir / img for img in images]

# Copy originals to report/images
for img_path in image_paths:
    shutil.copy(img_path, report_img_dir / img_path.name)

# Initialize EasyOCR (GPU=False for CPU)
reader = easyocr.Reader(['en'], gpu=False)

results = {}

for img_path in image_paths:
    img_name = img_path.name
    print(f'Processing {img_name}')
    
    # Load image
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # OCR
    ocr_results = reader.readtext(str(img_path))
    
    bboxes = []
    texts = []
    confidences = []
    bbox_areas = []
    
    for (bbox, text, conf) in ocr_results:
        pts = np.array(bbox, dtype=np.int32)
        area = cv2.contourArea(pts)
        bbox_areas.append(area)
        bboxes.append(bbox)
        texts.append(text)
        confidences.append(conf)
    
    # Filter small bboxes: < 1% of image area
    img_area = h * w
    small_mask = np.array(bbox_areas) < 0.01 * img_area
    small_bboxes = [bboxes[i] for i in range(len(bboxes)) if small_mask[i]]
    small_texts = [texts[i] for i in range(len(texts)) if small_mask[i]]
    small_confs = [confidences[i] for i in range(len(confidences)) if small_mask[i]]
    
    results[img_name] = {
        'total_detections': len(ocr_results),
        'small_detections': len(small_bboxes),
        'img_shape': (h, w),
        'small_bboxes': small_bboxes,
        'small_texts': small_texts,
        'small_confs': small_confs
    }
    
    # Save bboxes json
    bbox_data = []
    for i, bbox in enumerate(small_bboxes):
        bbox_data.append({
            'bbox': bbox,
            'text': small_texts[i],
            'conf': small_confs[i]
        })
    with open(output_dir / f'{img_name}_small_bboxes.json', 'w') as f:
        json.dump(bbox_data, f, indent=2)
    
    # Visualize bbox overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    for bbox in small_bboxes:
        pts = np.array(bbox, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_rgb, [pts], True, (0, 255, 0), 3)
    plt.title(f'Small Text ROIs in {img_name}')
    plt.axis('off')
    plt.savefig(report_img_dir / f'{img_name}_bbox_overlay.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Crop and zoom small regions
    crop_dir = output_dir / f'{img_name}_crops'
    crop_dir.mkdir(exist_ok=True)
    for i, bbox in enumerate(small_bboxes[:5]):  # Top 5
        pts = np.int32(bbox)
        x, y, w_crop, h_crop = cv2.boundingRect(pts)
        crop = img[y:y+h_crop, x:x+w_crop]
        crop_resized = cv2.resize(crop, (224, 224))
        cv2.imwrite(str(crop_dir / f'crop_{i}.png'), crop_resized)
        # Also save overlay crop
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(5,5))
        plt.imshow(crop_rgb)
        plt.title(f'Crop {i}: {small_texts[i][:20]}')
        plt.axis('off')
        plt.savefig(report_img_dir / f'{img_name}_crop_{i}.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    # Histogram of bbox sizes
    if bbox_areas:
        plt.figure(figsize=(10, 6))
        sns.histplot(bbox_areas, bins=30, kde=True)
        plt.axvline(0.01 * img_area, color='r', linestyle='--', label='1% threshold')
        plt.xlabel('BBox Area (pixels)')
        plt.ylabel('Count')
        plt.title(f'BBox Size Distribution - {img_name}')
        plt.legend()
        plt.savefig(report_img_dir / f'{img_name}_bbox_hist.png', bbox_inches='tight', dpi=150)
        plt.close()

# Summary table
df_data = []
for img_name, res in results.items():
    df_data.append({
        'Image': img_name,
        'Height': res['img_shape'][0],
        'Width': res['img_shape'][1],
        'Total Detections': res['total_detections'],
        'Small ROIs (<1%)': res['small_detections']
    })
df = pd.DataFrame(df_data)
df.to_csv(output_dir / 'detection_summary.csv', index=False)
fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.savefig(report_img_dir / 'detection_summary_table.png', bbox_inches='tight', dpi=150)
plt.close()

# Save full results
with open(output_dir / 'full_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Analysis complete. Check outputs/ and report/images/')
