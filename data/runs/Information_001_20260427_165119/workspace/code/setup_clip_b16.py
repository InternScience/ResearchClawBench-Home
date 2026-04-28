import os, time, sys
os.environ['HF_HOME']='/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/hfcache'
os.environ['TORCH_HOME']='/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/torchcache'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='240'
import open_clip, torch
t0=time.time()
# B/16 - 224
model, _, prepro = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
tok = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
print('load_time', time.time()-t0, flush=True)
img = torch.randn(1,3,224,224)
with torch.no_grad():
    f = model.encode_image(img)
print('feat shape', tuple(f.shape), flush=True)
