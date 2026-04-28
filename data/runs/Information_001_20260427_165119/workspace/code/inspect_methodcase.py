"""Quick visual inspection of relevancy maps and crops."""
import os, sys, json, numpy as np
sys.path.insert(0, '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/code')
from PIL import Image
from vicrop import VicropModel, relevancy_to_bbox

WS = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119'
vm = VicropModel()

# bookshop clock
pil = Image.open(os.path.join(WS, 'data/demo_imgs/method_case.png')).convert('RGB')
print('Full image size:', pil.size)
# Note: method_case.png is a composite image with the clock case in upper-left.
# Crop to that part first
W, H = pil.size
upper = pil.crop((0, 0, W//2, H//2))
print('upper-left region size:', upper.size)
upper.save(os.path.join(WS, 'outputs', 'method_case_upper_left.png'))

rel = vm.relevancy(upper, 'a clock with numbers on a wall above a bookshop window')
print('relevancy max', rel['chefer'].max(), 'mean', rel['chefer'].mean())
print('argmax', np.unravel_index(np.argmax(rel['chefer']), rel['chefer'].shape))
bbox = relevancy_to_bbox(rel['chefer'], upper.size, threshold_pct=0.85, margin=0.05)
print('bbox:', bbox)
