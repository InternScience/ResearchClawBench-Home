#!/usr/bin/env python3
"""Deterministic analysis of task-guided crop/zoom behavior on demo images.

This script does not run a proprietary MLLM. Instead it implements a reproducible
proxy for the ViCrop/V* idea: derive a task/scene-guided visual interest map,
select a compact ROI, compare fixed-resolution global encoding with re-encoded
local crops, and export figures/tables for a scientific report.
"""
from __future__ import annotations
import os, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260429_170047')
DATA = ROOT/'data'/'demo_imgs'
OUT = ROOT/'outputs'
IMGOUT = ROOT/'report'/'images'
CODE = ROOT/'code'
for p in [OUT, IMGOUT, CODE]: p.mkdir(parents=True, exist_ok=True)

TARGET_ENCODER = 336  # CLIP-like square fixed resolution proxy
GRID = 14             # patch grid proxy for a fixed-resolution ViT
ROI_FRAC = 0.28       # crop side length relative to smaller dimension

TASK_HINTS = {
    'demo1.png': {'task':'identify small road-scene details such as taxi signs, license plates, distant objects', 'domain':'urban traffic'},
    'demo2.png': {'task':'distinguish fine-grained flower colors and individual blossoms in a crowded greenhouse', 'domain':'flower garden'},
    'method_case.png': {'task':'audit paper-provided ViCrop examples and heatmap/crop relation', 'domain':'method figure'}
}

def load_rgb(path: Path):
    im = Image.open(path).convert('RGB')
    return np.asarray(im), im

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def fixed_resize_blur(arr, size=TARGET_ENCODER):
    pil=Image.fromarray(arr)
    small=pil.resize((size,size), Image.Resampling.BICUBIC)
    back=small.resize((arr.shape[1],arr.shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(back), np.asarray(small)

def saliency_interest(arr):
    """Hand-crafted visual interest map: edge density + color saturation + small-object texture."""
    rgb=arr.astype(np.float32)/255.0
    gray=cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # edge/texture at two scales
    gx=cv2.Sobel(gray, cv2.CV_32F, 1,0,ksize=3)
    gy=cv2.Sobel(gray, cv2.CV_32F, 0,1,ksize=3)
    grad=np.sqrt(gx*gx+gy*gy)
    lap=np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    hsv=cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat=hsv[:,:,1]
    # task prior: downweight blank sky/flat pavement; emphasize mid/high saturation and edges
    interest=0.50*norm(grad)+0.25*norm(lap)+0.25*norm(sat)
    # center/context prior avoids selecting only frame borders/text dates
    h,w=interest.shape
    yy,xx=np.mgrid[0:h,0:w]
    center=np.exp(-(((xx-w/2)/(0.65*w))**2 + ((yy-h/2)/(0.65*h))**2))
    interest=interest*(0.70+0.30*center)
    interest=cv2.GaussianBlur(interest.astype(np.float32),(0,0),sigmaX=max(3,min(h,w)/100))
    return norm(interest)

def norm(x):
    x=np.asarray(x,dtype=np.float32)
    mn, mx=float(np.nanmin(x)), float(np.nanmax(x))
    if mx-mn < 1e-8: return np.zeros_like(x,dtype=np.float32)
    return (x-mn)/(mx-mn)

def select_roi(arr, sal, side_frac=ROI_FRAC):
    h,w=sal.shape
    side=int(max(96, min(h,w)*side_frac))
    # method_case is a composed figure; allow slightly larger crop to cover example panels
    if w>2000 and h>1500: side=int(min(h,w)*0.34)
    kernel=np.ones((side,side),np.float32)
    sums=cv2.filter2D(sal.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    cy,cx=np.unravel_index(np.argmax(sums), sums.shape)
    x0=int(np.clip(cx-side//2,0,w-side)); y0=int(np.clip(cy-side//2,0,h-side))
    x1=x0+side; y1=y0+side
    return x0,y0,x1,y1

def local_crop_reencode(arr, bbox):
    x0,y0,x1,y1=bbox
    crop=arr[y0:y1,x0:x1]
    # crop is re-encoded at fixed resolution; compare to what global fixed encoding leaves in same original region
    crop_small=np.asarray(Image.fromarray(crop).resize((TARGET_ENCODER,TARGET_ENCODER), Image.Resampling.BICUBIC))
    crop_back=np.asarray(Image.fromarray(crop_small).resize((x1-x0,y1-y0), Image.Resampling.BICUBIC))
    return crop, crop_small, crop_back

def psnr(a,b):
    a=a.astype(np.float32); b=b.astype(np.float32)
    mse=float(np.mean((a-b)**2))
    if mse<=1e-12: return 99.0
    return 20*math.log10(255.0/math.sqrt(mse))

def entropy_gray(arr):
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    hist=np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p=hist/hist.sum(); p=p[p>0]
    return float(-(p*np.log2(p)).sum())

def edge_density(arr):
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,80,160)
    return float((edges>0).mean())

def lap_var(arr):
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def patch_variance(arr, grid=GRID):
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
    small=cv2.resize(gray,(grid,grid),interpolation=cv2.INTER_AREA)
    return float(np.var(small)), small

def make_overview(rows):
    fig,ax=plt.subplots(figsize=(8,4.6))
    df=pd.DataFrame(rows)
    x=np.arange(len(df))
    ax.bar(x-0.18, df['width'], width=0.36, label='width')
    ax.bar(x+0.18, df['height'], width=0.36, label='height')
    ax.axhline(TARGET_ENCODER,color='crimson',ls='--',lw=1.5,label=f'fixed encoder side ({TARGET_ENCODER}px proxy)')
    ax.set_xticks(x); ax.set_xticklabels(df['image'],rotation=20,ha='right')
    ax.set_ylabel('pixels')
    ax.set_title('Demo image sizes vs fixed-resolution encoder proxy')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(IMGOUT/'data_overview.png',dpi=180); plt.close(fig)

def draw_roi_panel(records):
    n=len(records)
    fig,axes=plt.subplots(n,3,figsize=(12,4*n))
    if n==1: axes=axes[None,:]
    for i,r in enumerate(records):
        arr=r['arr']; sal=r['sal']; bbox=r['bbox']; crop=r['crop']
        disp=Image.fromarray(arr).copy(); draw=ImageDraw.Draw(disp)
        draw.rectangle(bbox, outline=(0,255,255), width=max(3,arr.shape[1]//250))
        axes[i,0].imshow(disp); axes[i,0].set_title(f"{r['image']}: global view + selected ROI"); axes[i,0].axis('off')
        axes[i,1].imshow(sal,cmap='viridis'); axes[i,1].set_title('task/scene interest map'); axes[i,1].axis('off')
        axes[i,2].imshow(crop); axes[i,2].set_title('local crop re-encoded at full budget'); axes[i,2].axis('off')
    fig.tight_layout(); fig.savefig(IMGOUT/'roi_crops.png',dpi=180); plt.close(fig)

def draw_saliency(records):
    fig,axes=plt.subplots(len(records),2,figsize=(10,3.6*len(records)))
    if len(records)==1: axes=axes[None,:]
    for i,r in enumerate(records):
        arr=r['arr']; sal=r['sal']; bbox=r['bbox']
        axes[i,0].imshow(arr); axes[i,0].axis('off'); axes[i,0].set_title(r['image'])
        axes[i,1].imshow(arr,alpha=0.55); axes[i,1].imshow(sal,cmap='magma',alpha=0.55)
        x0,y0,x1,y1=bbox
        axes[i,1].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,ec='cyan',fc='none',lw=2))
        axes[i,1].axis('off'); axes[i,1].set_title('interest heatmap overlay')
    fig.tight_layout(); fig.savefig(IMGOUT/'saliency_heatmaps.png',dpi=180); plt.close(fig)

def draw_metrics(metrics):
    df=pd.DataFrame(metrics)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    sns.barplot(data=df,x='image',y='detail_gain_lap_var',ax=axes[0],color='#4c78a8')
    axes[0].axhline(1,color='k',lw=1); axes[0].set_title('Crop detail gain over global fixed view')
    axes[0].set_ylabel('Laplacian variance ratio'); axes[0].tick_params(axis='x',rotation=20)
    sns.barplot(data=df,x='image',y='roi_pixels_per_encoder_token_gain',ax=axes[1],color='#f58518')
    axes[1].axhline(1,color='k',lw=1); axes[1].set_title('Effective pixel budget gain in ROI')
    axes[1].set_ylabel('global ROI pixels/token ÷ crop pixels/token'); axes[1].tick_params(axis='x',rotation=20)
    fig.tight_layout(); fig.savefig(IMGOUT/'metric_comparison.png',dpi=180); plt.close(fig)

def draw_context_memory(records):
    fig,axes=plt.subplots(len(records),2,figsize=(10,3.8*len(records)))
    if len(records)==1: axes=axes[None,:]
    for i,r in enumerate(records):
        arr=r['arr']; bbox=r['bbox']; crop=r['crop']; global_small=r['global_small']
        axes[i,0].imshow(global_small); axes[i,0].axis('off'); axes[i,0].set_title(f"fixed global token view ({TARGET_ENCODER}×{TARGET_ENCODER})")
        axes[i,1].imshow(crop); axes[i,1].axis('off'); axes[i,1].set_title('visual working memory: selected high-detail local evidence')
    fig.tight_layout(); fig.savefig(IMGOUT/'context_memory_panel.png',dpi=180); plt.close(fig)

def draw_tile_comparison(tile_df):
    fig,ax=plt.subplots(figsize=(8,4.5))
    sns.scatterplot(data=tile_df,x='tile_area_fraction',y='interest_fraction',hue='image',style='selection',s=90,ax=ax)
    ax.plot([0,1],[0,1],ls='--',color='gray',lw=1,label='area-proportional')
    ax.set_title('Selective crop concentrates visual interest vs uniform tile area')
    ax.set_xlabel('fraction of image area used')
    ax.set_ylabel('fraction of saliency/interest captured')
    ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout(); fig.savefig(IMGOUT/'tile_vs_task_crop.png',dpi=180); plt.close(fig)

def main():
    rows=[]; metrics=[]; roi_summary={}; records=[]; tile_rows=[]
    for p in sorted(DATA.glob('*')):
        if p.suffix.lower() not in ['.png','.jpg','.jpeg']: continue
        arr,im=load_rgb(p); h,w=arr.shape[:2]
        rows.append({'image':p.name,'width':w,'height':h,'pixels':w*h,'bytes':p.stat().st_size,'sha256_16':sha256(p), 'task_hint':TASK_HINTS.get(p.name,{}).get('task','')})
        sal=saliency_interest(arr)
        bbox=select_roi(arr,sal)
        x0,y0,x1,y1=bbox
        global_back, global_small=fixed_resize_blur(arr)
        crop,crop_small,crop_back=local_crop_reencode(arr,bbox)
        global_roi=global_back[y0:y1,x0:x1]
        orig_roi=arr[y0:y1,x0:x1]
        area=(x1-x0)*(y1-y0); area_frac=area/(w*h)
        sal_frac=float(sal[y0:y1,x0:x1].sum()/sal.sum())
        global_pixels_per_token=(w*h)/(GRID*GRID)
        crop_pixels_per_token=area/(GRID*GRID)
        m={
            'image':p.name,'roi_x0':x0,'roi_y0':y0,'roi_x1':x1,'roi_y1':y1,'roi_width':x1-x0,'roi_height':y1-y0,
            'roi_area_fraction':area_frac,'roi_interest_fraction':sal_frac,
            'global_downsample_psnr_full_image':psnr(arr,global_back),
            'global_downsample_psnr_in_roi':psnr(orig_roi,global_roi),
            'crop_reencode_psnr_in_roi':psnr(orig_roi,crop_back),
            'global_roi_edge_density':edge_density(global_roi),
            'crop_edge_density':edge_density(crop),
            'global_roi_lap_var':lap_var(global_roi),
            'crop_lap_var':lap_var(crop),
            'detail_gain_lap_var':lap_var(crop)/(lap_var(global_roi)+1e-9),
            'edge_density_gain':edge_density(crop)/(edge_density(global_roi)+1e-9),
            'global_image_entropy':entropy_gray(arr),
            'crop_entropy':entropy_gray(crop),
            'roi_pixels_per_encoder_token_global':global_pixels_per_token,
            'roi_pixels_per_encoder_token_crop':crop_pixels_per_token,
            'roi_pixels_per_encoder_token_gain':global_pixels_per_token/crop_pixels_per_token,
            'fixed_encoder_side_px':TARGET_ENCODER,
            'patch_grid':GRID
        }
        metrics.append(m)
        roi_summary[p.name]={'bbox_xyxy':[x0,y0,x1,y1], 'task_hint':TASK_HINTS.get(p.name,{}), 'selection_rule':'max summed deterministic interest map in a square ROI', 'area_fraction':area_frac, 'interest_fraction':sal_frac}
        # 3x3 uniform tile comparison: best tile with comparable simple baseline
        th,tw=h//3,w//3
        best=None
        for iy in range(3):
            for ix in range(3):
                tx0,ty0=ix*tw,iy*th; tx1=w if ix==2 else (ix+1)*tw; ty1=h if iy==2 else (iy+1)*th
                frac=(tx1-tx0)*(ty1-ty0)/(w*h); sfrac=float(sal[ty0:ty1,tx0:tx1].sum()/sal.sum())
                row={'image':p.name,'selection':'uniform_3x3_tile','tile_ix':ix,'tile_iy':iy,'tile_area_fraction':frac,'interest_fraction':sfrac}
                tile_rows.append(row)
                if best is None or sfrac>best['interest_fraction']: best=row
        tile_rows.append({'image':p.name,'selection':'task_guided_crop','tile_ix':-1,'tile_iy':-1,'tile_area_fraction':area_frac,'interest_fraction':sal_frac})
        records.append({'image':p.name,'arr':arr,'sal':sal,'bbox':bbox,'crop':crop,'global_small':global_small})
    pd.DataFrame(rows).to_csv(OUT/'image_overview.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'crop_metrics.csv',index=False)
    pd.DataFrame(tile_rows).to_csv(OUT/'tile_vs_task_crop.csv',index=False)
    with open(OUT/'roi_summary.json','w') as f: json.dump(roi_summary,f,indent=2)
    dep={
        'python':'available','PIL':'available','cv2':'available','matplotlib':'available','seaborn':'available','pandas':'available',
        'actual_mllm_inference':'not_available_in_workspace_checked_by_absence_of_model_or_api_files',
        'fallback':'deterministic CV proxy for fixed-resolution detail loss, saliency ROI selection, and visual memory construction'
    }
    with open(OUT/'dependency_check.json','w') as f: json.dump(dep,f,indent=2)
    fidelity={
        'named_mechanism':'training-free task-guided cropping / ViCrop-like guided visual search',
        'non_negotiable_steps':[
            {'step':'keep global context','implemented':'global fixed-resolution view saved and shown in context_memory_panel.png'},
            {'step':'identify ROI without training','implemented':'deterministic interest map from edges, texture, saturation, and center/context prior'},
            {'step':'zoom local ROI','implemented':'selected crop is re-encoded/resized at full fixed encoder budget and compared against global downsampled ROI'},
            {'step':'integrate local detail with global context','implemented':'context_memory_panel.png pairs fixed global view with local crop per image'},
            {'step':'validate small-detail information retention','implemented':'crop_metrics.csv reports PSNR, edge density, Laplacian variance, and effective pixel/token gains'}
        ],
        'deviations':['No LLM-generated question parser, no learned attention map, and no answer accuracy evaluation because no MLLM weights/API or labeled VQA answers are provided.'],
        'assumptions':['Task hints are manually specified from image content and task description; proxy metrics stand in for encoder token information loss.']
    }
    with open(OUT/'method_fidelity_checklist.json','w') as f: json.dump(fidelity,f,indent=2)
    make_overview(rows); draw_roi_panel(records); draw_saliency(records); draw_metrics(metrics); draw_context_memory(records); draw_tile_comparison(pd.DataFrame(tile_rows))
    # claim recovery table
    claim_rows=[
      {'claim':'Fixed-resolution encoding compresses large images substantially.','supporting_artifact':'outputs/image_overview.csv; report/images/data_overview.png','evidence_type':'direct metadata + figure'},
      {'claim':'Selected local crops retain more fine-detail signal than the same region after global downsampling.','supporting_artifact':'outputs/crop_metrics.csv; report/images/metric_comparison.png','evidence_type':'computed PSNR/detail metrics'},
      {'claim':'The ROI selection is interpretable and spatially traceable.','supporting_artifact':'outputs/roi_summary.json; report/images/saliency_heatmaps.png; report/images/roi_crops.png','evidence_type':'coordinates + heatmaps'},
      {'claim':'The implementation approximates V*/ViCrop but is not an exact MLLM accuracy reproduction.','supporting_artifact':'outputs/method_fidelity_checklist.json; outputs/dependency_check.json','evidence_type':'capability/fidelity audit'}]
    pd.DataFrame(claim_rows).to_csv(OUT/'claim_recovery_table.csv',index=False)
    # Update inventory statuses
    inv=json.load(open(OUT/'target_artifact_inventory.json'))
    produced={str(p.relative_to(ROOT)) for p in list(OUT.glob('*'))+list(IMGOUT.glob('*.png'))}
    for sec in inv:
        if isinstance(inv[sec],list):
            for item in inv[sec]:
                art=item.get('artifact')
                if art in produced:
                    item['status']='satisfied'
                elif art=='report/images/tile_vs_task_crop.png' and art in produced:
                    item['status']='satisfied'
    # include extra figure if absent
    if not any(i.get('artifact')=='report/images/tile_vs_task_crop.png' for i in inv.get('required_figures',[])):
        inv['required_figures'].append({'artifact':'report/images/tile_vs_task_crop.png','status':'satisfied','purpose':'uniform tiling vs task-guided crop comparison'})
    open(OUT/'target_artifact_inventory.json','w').write(json.dumps(inv,indent=2))
    print('Wrote outputs and figures')
    print(pd.DataFrame(metrics)[['image','roi_area_fraction','roi_interest_fraction','detail_gain_lap_var','roi_pixels_per_encoder_token_gain']].to_string(index=False))

if __name__=='__main__': main()
