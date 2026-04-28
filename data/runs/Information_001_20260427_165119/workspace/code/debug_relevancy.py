"""Quick debug - check relevancy maps and bbox on demo images."""
import os, sys, json, numpy as np
sys.path.insert(0, '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/code')
from vicrop import VicropModel, vicrop_predict, relevancy_to_bbox
from PIL import Image

WS = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119'

print('Loading...', flush=True)
vm = VicropModel()
print('Loaded.', flush=True)

cases = [
    ('demo1.png', 'a license plate of a yellow taxi'),
    ('demo2.png', 'small yellow tulip flowers'),
]
for fn, q in cases:
    pil = Image.open(os.path.join(WS, 'data/demo_imgs', fn)).convert('RGB')
    print(f'\n== {fn}  size={pil.size}  query={q!r} ==', flush=True)
    rel = vm.relevancy(pil, q)
    chefer = rel['chefer']
    print('chefer min/max/mean', chefer.min(), chefer.max(), chefer.mean())
    print('rollout min/max/mean', rel['rollout'].min(), rel['rollout'].max(), rel['rollout'].mean())
    print('argmax (i,j) chefer:', np.unravel_index(np.argmax(chefer), chefer.shape))
    bbox = relevancy_to_bbox(chefer, pil.size, threshold_pct=0.85, margin=0.05)
    print('bbox @0.85:', bbox)
    bbox2 = relevancy_to_bbox(chefer, pil.size, threshold_pct=0.7, margin=0.10)
    print('bbox @0.70:', bbox2)
