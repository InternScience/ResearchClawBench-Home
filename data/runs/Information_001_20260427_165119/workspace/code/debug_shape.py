import os, sys, torch
sys.path.insert(0, '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/code')
from vicrop import VicropModel
from PIL import Image
vm = VicropModel()
pil = Image.open('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/data/demo_imgs/demo1.png').convert('RGB')
x = vm.prepro(pil).unsqueeze(0)
out = vm.model.encode_image(x)
print('captured 0 attn shape:', vm.captured[0].attn_weights.shape)
print('captured -1 attn shape:', vm.captured[-1].attn_weights.shape)
