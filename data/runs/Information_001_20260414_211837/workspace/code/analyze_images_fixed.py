import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import shutil
import warnings
warnings.filterwarnings('ignore')

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
    shutil.copy2(img_path, report_img_dir / img_path.name)

def bbox_to_serial(bbox):
    return [[int(float(p[0])), int(float(p[1]))] for p in bbox]

# Initialize EasyOCR (GPU=False for CPU)
print('Initializing EasyOCR...')
reader = easyocr.Reader(['en'], gpu=False)

results = {}

for img_path in image_paths:
    img_name = img_path.name[:-4]  # remove .png
    print(f'Processing {img_name}')
    
    # Load image
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    img_area = h * w
    
    # OCR
    ocr_results = reader.readtext(str(img_path))
    print(f'Total detections: {len(ocr_results)}')
    
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
        confidences.append(float(conf))
    
    # Filter small bboxes: < 1% of image area
    small_mask = np.array(bbox_areas) < 0.01 * img_area
    small_idx = np.where(small_mask)[0]
    small_bboxes = [bboxes[i] for i in small_idx]
    small_texts = [texts[i] for i in small_idx]
    small_confs = [confidences[i] for i in small_idx]
    
    num_small = len(small_bboxes)
    print(f'Small detections: {num_small}')
    
    results[img_name] = {
        'total_detections': len(ocr_results),
        'small_detections': num_small,
        'img_shape': (h, w),
        'small_bboxes_ser': [bbox_to_serial(b) for b in small_bboxes],
        'small_texts': small_texts,
        'small_confs': small_confs
    }
    
    # Save bboxes json
    bbox_data = []
    for i in range(num_small):
        bbox_data.append({
            'bbox': results[img_name]['small_bboxes_ser'][i],
            'text': small_texts[i],
            'conf': small_confs[i]
        })
    with open(output_dir / f'{img_name}_small_bboxes.json', 'w') as f:
        json.dump(bbox_data, f, indent=2)
    
    # Visualize bbox overlay
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    for b in small_bboxes:
        pts = np.array(b, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img_display, [pts], True, (0,255,0), 3)
    
    plt.figure(figsize=(15,10))
    plt.imshow(img_display)
    plt.title(f'Small Text ROIs (<1% area) in {img_name} ({num_small} ROIs)', fontsize=16)
    plt.axis('off')
    plt.savefig(report_img_dir / f'{img_name}_bbox_overlay.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    # Histogram if any detections
    if bbox_areas:
        plt.figure(figsize=(10,6))
        sns.histplot(bbox_areas, bins=min(30, len(bbox_areas)), kde=True)
        plt.axvline(0.01 * img_area, color='r', ls='--', label='1% threshold')
        plt.xlabel('BBox Area (pixels)')
        plt.ylabel('Frequency')
        plt.title(f'BBox Area Distribution - {img_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_img_dir / f'{img_name}_bbox_hist.png', dpi=150)
        plt.close()
    
    # Top 5 crops
    crop_dir = output_dir / f'{img_name}_crops'
    crop_dir.mkdir(exist_ok=True)
    for i in range(min(5, num_small)):
        bbox = small_bboxes[i]
        pts = np.int32(bbox)
        x,y,w_c,h_c = cv2.boundingRect(pts)
        crop = img[y:y+h_c, x:x+w_c]
        crop_resized = cv2.resize(crop, (224,224))
        cv2.imwrite(str(crop_dir / f'crop_{i:02d}.png'), crop_resized)
        
        # Plot crop
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,6))
        plt.imshow(crop_rgb)
        plt.title(f'Zoom Crop {i}: \"{small_texts[i][:30]}...\" (conf: {small_confs[i]:.2f})')
        plt.axis('off')
        plt.savefig(report_img_dir / f'{img_name}_crop_{i:02d}.png', bbox_inches='tight', dpi=150)
        plt.close()

# Summary table
df_data = []
for img_name, res in results.items():
    df_data.append({
        'Image': img_name,
        'Resolution': f"{res['img_shape'][1]}x{res['img_shape'][0]}",
        'Total Texts': res['total_detections'],
        'Small ROIs (<1%)': res['small_detections']
    })
df = pd.DataFrame(df_data)
df.to_csv(output_dir / 'detection_summary.csv', index=False)

fig, ax = plt.subplots(figsize=(10,3))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
plt.savefig(report_img_dir / 'detection_summary.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()

# Save full results
with open(output_dir / 'full_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Analysis complete!')