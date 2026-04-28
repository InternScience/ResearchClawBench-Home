"""Run the full ViCrop experiment on the demo images and save all artifacts.

Tasks are designed around fine-grained questions where the target object
covers a small fraction of the image — exactly the regime ViCrop is meant
to help with.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom as ndzoom

sys.path.insert(0, '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/code')
from vicrop import (VicropModel, vicrop_predict, baseline_predict,
                    relevancy_to_bbox, softmax)

WS = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119'
IMG_DIR = os.path.join(WS, 'data/demo_imgs')
OUT_DIR = os.path.join(WS, 'outputs')
FIG_DIR = os.path.join(WS, 'report/images')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# Each task has:
#   - image:       file in data/demo_imgs/
#   - question:    the natural-language VQA question
#   - query_text:  text used to compute the task-conditioned relevancy map
#                  (this is what the "task-guided" cropping step is conditioned on)
#   - options:     full natural-language options scored by CLIP
#   - short_options: short labels for plotting
#   - gold_idx:    index of the ground-truth option (verified from the image)
TASKS = [
    {  # T0
        'image': 'demo1.png',
        'question': "What is the first letter of the silver car's licence plate?",
        'query_text': 'a licence plate on the front of a silver sedan car',
        'options': [
            'a licence plate that starts with the letter R',
            'a licence plate that starts with the letter T',
            'a licence plate that starts with the letter A',
            'a licence plate that starts with the letter B',
        ],
        'short_options': ['R', 'T', 'A', 'B'],
        'gold_idx': 0,
        'short_question': "Plate first letter",
    },
    {  # T1
        'image': 'demo1.png',
        'question': "Are the police officers wearing helmets?",
        'query_text': 'a police officer wearing a blue helmet',
        'options': [
            'two police officers wearing blue helmets',
            'two police officers wearing red helmets',
            'two police officers without helmets',
            'two police officers wearing yellow helmets',
        ],
        'short_options': ['blue', 'red', 'no helmet', 'yellow'],
        'gold_idx': 0,
        'short_question': "Police helmet colour",
    },
    {  # T2
        'image': 'demo1.png',
        'question': "What sign is on the building above the shop?",
        'query_text': 'a red and white shop sign with text on the front of a building',
        'options': [
            'a shop sign that reads DAVIVIENDA',
            'a shop sign that reads STARBUCKS',
            'a shop sign that reads MCDONALDS',
            'a shop sign that reads WALMART',
        ],
        'short_options': ['DAVIVIENDA', 'STARBUCKS', 'MCDONALDS', 'WALMART'],
        'gold_idx': 0,
        'short_question': "Building shop sign",
    },
    {  # T3
        'image': 'demo2.png',
        'question': "What colour are the tulips in the bottom-right corner of the greenhouse?",
        'query_text': 'a cluster of tulip flowers in the lower right corner of the photo',
        'options': [
            'yellow tulip flowers',
            'red tulip flowers',
            'pink tulip flowers',
            'purple tulip flowers',
        ],
        'short_options': ['yellow', 'red', 'pink', 'purple'],
        'gold_idx': 0,
        'short_question': "Bottom-right tulip colour",
    },
    {  # T4
        'image': 'demo2.png',
        'question': "What colour are the tulips on the far left edge of the central display?",
        'query_text': 'tulip flowers on the far left edge of the central display',
        'options': [
            'magenta or hot pink tulips',
            'red tulips',
            'white tulips',
            'orange tulips',
        ],
        'short_options': ['magenta', 'red', 'white', 'orange'],
        'gold_idx': 0,
        'short_question': "Far-left tulip colour",
    },
    {  # T5
        'image': 'demo2.png',
        'question': "What colour are the tulips at the very front of the central display?",
        'query_text': 'tulip flowers at the very front of the central display',
        'options': [
            'red tulips',
            'yellow tulips',
            'white tulips',
            'pink tulips',
        ],
        'short_options': ['red', 'yellow', 'white', 'pink'],
        'gold_idx': 0,
        'short_question': "Front-row tulip colour",
    },
]


def overlay_relevancy(pil: Image.Image, rel_map: np.ndarray, alpha=0.50):
    arr = np.array(pil).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    g = rel_map.shape[0]
    up = ndzoom(rel_map, (H / g, W / g), order=1)
    if up.max() > 0:
        up = (up - up.min()) / (up.max() - up.min() + 1e-9)
    cmap = plt.get_cmap('jet')(up)[..., :3]
    blend = (1 - alpha) * arr + alpha * cmap
    return np.clip(blend, 0, 1)


def run():
    print('Loading CLIP...', flush=True)
    vm = VicropModel()
    print('Loaded.', flush=True)

    rng = np.random.default_rng(42)
    all_results = []
    relevancy_maps = []  # for figure
    for ti, T in enumerate(TASKS):
        print(f'\n=== T{ti}: {T["short_question"]} ({T["image"]}) ===', flush=True)
        pil = Image.open(os.path.join(IMG_DIR, T['image'])).convert('RGB')
        # ViCrop main
        vc = vicrop_predict(vm, pil, T['query_text'], T['options'],
                            threshold_pct=0.85, margin=0.05, w_crop=0.5)
        # Baselines
        b_nocrop = baseline_predict(vm, pil, T['options'], 'nocrop')
        b_center = baseline_predict(vm, pil, T['options'], 'center', frac=0.5)
        b_random = baseline_predict(vm, pil, T['options'], 'random', frac=0.5,
                                     seed=int(rng.integers(0, 1e6)))
        b_tile = baseline_predict(vm, pil, T['options'], 'tile')
        gold = T['gold_idx']

        result = {
            'task_idx': ti,
            'image': T['image'],
            'question': T['question'],
            'short_question': T['short_question'],
            'gold_option': T['short_options'][gold],
            'gold_idx': gold,
            'short_options': T['short_options'],
            'image_size': list(pil.size),
            'bbox': list(vc['bbox']),
            'sims_global': vc['sims_global'],
            'sims_crop': vc['sims_crop'],
            'p_global': vc['p_global'],
            'p_crop': vc['p_crop'],
            'p_ens': vc['p_ens'],
            'pred_global_idx': int(vc['pred_global']),
            'pred_crop_idx': int(vc['pred_crop']),
            'pred_vicrop_idx': int(vc['pred_ens']),
            'baselines': {
                'nocrop_pred': int(b_nocrop['pred']),
                'center_pred': int(b_center['pred']),
                'random_pred': int(b_random['pred']),
                'tile_pred': int(b_tile['pred']),
            },
            'p_correct_vicrop': float(vc['p_ens'][gold]),
            'p_correct_global': float(vc['p_global'][gold]),
            'p_correct_crop': float(vc['p_crop'][gold]),
        }
        all_results.append(result)
        relevancy_maps.append({'chefer': vc['rel_chefer'],
                               'rollout': vc['rel_rollout']})
        np.save(os.path.join(OUT_DIR, f'rel_chefer_T{ti}.npy'), vc['rel_chefer'])
        np.save(os.path.join(OUT_DIR, f'rel_rollout_T{ti}.npy'), vc['rel_rollout'])
        gold_lbl = T['short_options'][gold]
        print(f"  gold = {gold_lbl}")
        print(f"  no-crop  = {T['short_options'][b_nocrop['pred']]}, "
              f"center = {T['short_options'][b_center['pred']]}, "
              f"random = {T['short_options'][b_random['pred']]}, "
              f"tile = {T['short_options'][b_tile['pred']]}")
        print(f"  ViCrop : global={T['short_options'][vc['pred_global']]}  "
              f"crop={T['short_options'][vc['pred_crop']]}  "
              f"ens={T['short_options'][vc['pred_ens']]}")
        print(f"  P(correct) global={vc['p_global'][gold]:.3f}, "
              f"crop={vc['p_crop'][gold]:.3f}, ens={vc['p_ens'][gold]:.3f}")
        print(f"  bbox = {vc['bbox']}")

    with open(os.path.join(OUT_DIR, 'per_demo_predictions.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    n = len(all_results)
    acc = {
        'no_crop': sum(r['baselines']['nocrop_pred'] == r['gold_idx'] for r in all_results) / n,
        'center_crop': sum(r['baselines']['center_pred'] == r['gold_idx'] for r in all_results) / n,
        'random_crop': sum(r['baselines']['random_pred'] == r['gold_idx'] for r in all_results) / n,
        'uniform_tile': sum(r['baselines']['tile_pred'] == r['gold_idx'] for r in all_results) / n,
        'vicrop_global_only': sum(r['pred_global_idx'] == r['gold_idx'] for r in all_results) / n,
        'vicrop_crop_only': sum(r['pred_crop_idx'] == r['gold_idx'] for r in all_results) / n,
        'vicrop_ensemble': sum(r['pred_vicrop_idx'] == r['gold_idx'] for r in all_results) / n,
    }
    p_correct = {
        'vicrop_global_only': float(np.mean([r['p_correct_global'] for r in all_results])),
        'vicrop_crop_only': float(np.mean([r['p_correct_crop'] for r in all_results])),
        'vicrop_ensemble': float(np.mean([r['p_correct_vicrop'] for r in all_results])),
    }
    main = {'accuracy': acc, 'mean_p_correct_for_correct_option': p_correct,
            'n_tasks': n}
    with open(os.path.join(OUT_DIR, 'main_results.json'), 'w') as f:
        json.dump(main, f, indent=2)
    print('\nMain results:', json.dumps(main, indent=2))

    # ---- Ablation on threshold and ensemble weight, on the same pre-computed
    # relevancy maps (re-using bbox for each threshold and recomputing crop
    # similarities only when the bbox changes).
    print('\n--- Ablation: threshold sweep ---', flush=True)
    thresholds = [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
    abl_thr = {'thresholds': thresholds, 'accuracy': []}
    for thr in thresholds:
        correct = 0
        for ti, T in enumerate(TASKS):
            pil = Image.open(os.path.join(IMG_DIR, T['image'])).convert('RGB')
            chefer = relevancy_maps[ti]['chefer']
            bbox = relevancy_to_bbox(chefer, pil.size, threshold_pct=thr, margin=0.05)
            sims_g = vm.score_options(pil, T['options'])
            sims_c = vm.score_options(pil.crop(bbox), T['options'])
            p = 0.5 * softmax(sims_g) + 0.5 * softmax(sims_c)
            if int(np.argmax(p)) == T['gold_idx']:
                correct += 1
        acc_thr = correct / len(TASKS)
        abl_thr['accuracy'].append(acc_thr)
        print(f"  threshold={thr:.2f}  acc={acc_thr:.3f}")

    print('\n--- Ablation: ensemble weight sweep ---', flush=True)
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    abl_w = {'w_crop': weights, 'accuracy': [], 'mean_p_correct': []}
    for w in weights:
        correct = 0
        pcorrect = []
        for ti, T in enumerate(TASKS):
            sg = np.array(all_results[ti]['p_global'])
            sc = np.array(all_results[ti]['p_crop'])
            p = (1 - w) * sg + w * sc
            if int(np.argmax(p)) == T['gold_idx']:
                correct += 1
            pcorrect.append(float(p[T['gold_idx']]))
        abl_w['accuracy'].append(correct / len(TASKS))
        abl_w['mean_p_correct'].append(float(np.mean(pcorrect)))
        print(f"  w_crop={w:.2f}  acc={correct/len(TASKS):.3f}  P(correct)={np.mean(pcorrect):.3f}")

    with open(os.path.join(OUT_DIR, 'ablations.json'), 'w') as f:
        json.dump({'threshold': abl_thr, 'w_crop': abl_w}, f, indent=2)

    return all_results, relevancy_maps


if __name__ == '__main__':
    run()
