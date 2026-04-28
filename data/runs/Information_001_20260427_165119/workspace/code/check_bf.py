import os
os.environ['HF_HOME']='/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/hfcache'
import open_clip
m,_,p = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
print('mha batch_first:', m.visual.transformer.resblocks[0].attn.batch_first)
