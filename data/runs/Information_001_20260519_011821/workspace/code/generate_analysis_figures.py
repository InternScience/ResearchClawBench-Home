"""
Generate additional analysis figures for the research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import json


def create_sensitivity_analysis_figure(save_path):
    """Figure showing MLLM sensitivity to visual concept size (from paper Table 1)."""
    size_categories = ['Small\n(S < 0.005)', 'Medium\n(0.005 ≤ S < 0.05)', 'Large\n(S ≥ 0.05)']
    x = np.arange(len(size_categories))
    width = 0.12

    # Data from Table 1
    blip2_no = [12.13, 19.57, 36.32]
    blip2_crop = [55.76, 52.02, 45.73]
    instruct_no = [21.79, 30.58, 45.30]
    instruct_crop = [69.60, 61.56, 53.39]
    llava_no = [39.38, 47.74, 50.65]
    llava_crop = [69.95, 65.36, 56.96]
    qwen_no = [56.42, 65.09, 68.60]
    qwen_crop = [70.35, 75.49, 71.05]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - 3.5*width, blip2_no, width, label='BLIP-2 (no crop)', color='#e74c3c', alpha=0.7)
    ax.bar(x - 2.5*width, blip2_crop, width, label='BLIP-2 (human-CROP)', color='#e74c3c', alpha=0.3, hatch='//')
    ax.bar(x - 1.5*width, instruct_no, width, label='InstructBLIP (no crop)', color='#3498db', alpha=0.7)
    ax.bar(x - 0.5*width, instruct_crop, width, label='InstructBLIP (human-CROP)', color='#3498db', alpha=0.3, hatch='//')
    ax.bar(x + 0.5*width, llava_no, width, label='LLaVA-1.5 (no crop)', color='#2ecc71', alpha=0.7)
    ax.bar(x + 1.5*width, llava_crop, width, label='LLaVA-1.5 (human-CROP)', color='#2ecc71', alpha=0.3, hatch='//')
    ax.bar(x + 2.5*width, qwen_no, width, label='Qwen-VL (no crop)', color='#9b59b6', alpha=0.7)
    ax.bar(x + 3.5*width, qwen_crop, width, label='Qwen-VL (human-CROP)', color='#9b59b6', alpha=0.3, hatch='//')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Visual Concept Size', fontsize=12)
    ax.set_title('MLLM Sensitivity to Visual Concept Size on TextVQA\n(Zhang et al., ICLR 2025)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_categories)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_pipeline_overview_figure(save_path):
    """Create a conceptual pipeline overview figure."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    # Step 1: Original Image + Question
    axes[0].text(0.5, 0.5, 'Original Image\n+\nQuestion', ha='center', va='center',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    axes[0].set_title('Step 1: Input', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Step 2: Importance Map
    axes[1].text(0.5, 0.5, 'Compute\nImportance Map\n(attention / gradient)', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    axes[1].set_title('Step 2: Localization', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Step 3: Crop Selection
    axes[2].text(0.5, 0.5, 'Sliding Window\nCrop Selection\n(max sum + edge contrast)', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    axes[2].set_title('Step 3: Crop', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # Step 4: Resize
    axes[3].text(0.5, 0.5, 'Crop Region\n+\nResize to Model Input', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    axes[3].set_title('Step 4: Zoom', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    # Step 5: Concatenate & Answer
    axes[4].text(0.5, 0.5, 'Concatenate\nOriginal + Cropped\nTokens → Answer', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.7))
    axes[4].set_title('Step 5: Enhanced VQA', fontsize=11, fontweight='bold')
    axes[4].axis('off')

    # Add arrows
    for i in range(4):
        fig.patches.append(patches.FancyArrowPatch(
            (axes[i].get_position().x1, axes[i].get_position().y0 + axes[i].get_position().height/2),
            (axes[i+1].get_position().x0, axes[i+1].get_position().y0 + axes[i+1].get_position().height/2),
            transform=fig.transFigure,
            arrowstyle='->', mutation_scale=30, linewidth=2, color='black'
        ))

    plt.suptitle('ViCrop Pipeline Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_demo_summary_figure(save_path):
    """Summary table of our demo experiment results."""
    with open('outputs/experiment_results.json', 'r') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.axis('tight')

    rows = []
    for r in results:
        img = r['image']
        q = r['question']
        baseline = r['answers']['baseline']
        rel = r['answers']['rel-att']
        grad = r['answers']['grad-att']
        pure = r['answers']['pure-grad']
        rows.append([img, q[:45] + '...' if len(q) > 45 else q, baseline, rel, grad, pure])

    table = ax.table(cellText=rows,
                     colLabels=['Image', 'Question', 'Baseline', 'rel-att', 'grad-att', 'pure-grad'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#4472C4']*6)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    for i in range(6):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('Demo Experiment Results: ViCrop on BLIP-VQA', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import matplotlib.patches as patches
    import os
    os.makedirs('report/images', exist_ok=True)

    create_sensitivity_analysis_figure('report/images/sensitivity_analysis.png')
    create_pipeline_overview_figure('report/images/pipeline_overview.png')
    create_demo_summary_figure('report/images/demo_summary_table.png')
    print("Analysis figures generated.")
