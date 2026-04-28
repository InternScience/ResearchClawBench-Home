"""Inspect the open_clip ViT-B/16 transformer to find attention layers for hooking."""
import os
os.environ['HF_HOME']='/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/hfcache'
os.environ['TORCH_HOME']='/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/torchcache'
import open_clip, torch
model,_,prepro = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
v = model.visual
print('VisualType', type(v).__name__)
print('image_size', getattr(v, 'image_size', None), 'patch_size', getattr(v, 'patch_size', None))
print('grid_size', getattr(v, 'grid_size', None))
# find transformer blocks
print('---- top-level children ----')
for n,c in v.named_children():
    print(n, type(c).__name__)
print('---- transformer.resblocks[0] ----')
trans = v.transformer
print(type(trans).__name__, 'num blocks', len(trans.resblocks))
b0 = trans.resblocks[0]
for n,c in b0.named_children():
    print(n, type(c).__name__)
print('---- attn module ----')
print(b0.attn)
