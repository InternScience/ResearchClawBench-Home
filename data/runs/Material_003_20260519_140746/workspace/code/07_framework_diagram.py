import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs('report/images', exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'AI-Guided Inverse Design Framework for Vitrimeric Polymers', 
        fontsize=16, fontweight='bold', ha='center', va='center')

# Colors
c_data = '#E8F4FD'
c_model = '#FFF4E6'
c_design = '#E8F8E8'
c_validate = '#FDE8E8'

# Block 1: Data & MD
box1 = FancyBboxPatch((0.3, 6.5), 3.2, 2.2, boxstyle="round,pad=0.1", 
                       facecolor=c_data, edgecolor='#2980B9', linewidth=2)
ax.add_patch(box1)
ax.text(1.9, 8.2, 'Molecular Dynamics', fontsize=11, fontweight='bold', ha='center', color='#2980B9')
ax.text(1.9, 7.8, '• Acid + Epoxide', fontsize=9, ha='center')
ax.text(1.9, 7.5, '• Tg simulation', fontsize=9, ha='center')
ax.text(1.9, 7.2, '• 8,424 vitrimer systems', fontsize=9, ha='center')
ax.text(1.9, 6.9, '• 295 calibration polymers', fontsize=9, ha='center')

# Block 2: GP Calibration
box2 = FancyBboxPatch((5.3, 6.5), 3.2, 2.2, boxstyle="round,pad=0.1", 
                       facecolor=c_model, edgecolor='#E67E22', linewidth=2)
ax.add_patch(box2)
ax.text(6.9, 8.2, 'GP Calibration', fontsize=11, fontweight='bold', ha='center', color='#E67E22')
ax.text(6.9, 7.8, '• RBF + White Kernel', fontsize=9, ha='center')
ax.text(6.9, 7.5, '• Morgan fingerprints', fontsize=9, ha='center')
ax.text(6.9, 7.2, '• MD → Experimental Tg', fontsize=9, ha='center')
ax.text(6.9, 6.9, '• Test R² = 0.857', fontsize=9, ha='center')

# Block 3: Graph VAE
box3 = FancyBboxPatch((10.3, 6.5), 3.2, 2.2, boxstyle="round,pad=0.1", 
                       facecolor=c_model, edgecolor='#8E44AD', linewidth=2)
ax.add_patch(box3)
ax.text(11.9, 8.2, 'Graph VAE', fontsize=11, fontweight='bold', ha='center', color='#8E44AD')
ax.text(11.9, 7.8, '• Dual encoder/decoder', fontsize=9, ha='center')
ax.text(11.9, 7.5, '• Latent dim = 32', fontsize=9, ha='center')
ax.text(11.9, 7.2, '• Property predictor', fontsize=9, ha='center')
ax.text(11.9, 6.9, '• Tg R² = 0.664', fontsize=9, ha='center')

# Block 4: Inverse Design
box4 = FancyBboxPatch((3.3, 3.5), 3.2, 2.2, boxstyle="round,pad=0.1", 
                       facecolor=c_design, edgecolor='#27AE60', linewidth=2)
ax.add_patch(box4)
ax.text(4.9, 5.2, 'Inverse Design', fontsize=11, fontweight='bold', ha='center', color='#27AE60')
ax.text(4.9, 4.8, '• Target Tg input', fontsize=9, ha='center')
ax.text(4.9, 4.5, '• Latent optimization', fontsize=9, ha='center')
ax.text(4.9, 4.2, '• Gradient descent', fontsize=9, ha='center')
ax.text(4.9, 3.9, '• Decode to molecules', fontsize=9, ha='center')

# Block 5: Validation
box5 = FancyBboxPatch((8.3, 3.5), 3.2, 2.2, boxstyle="round,pad=0.1", 
                       facecolor=c_validate, edgecolor='#C0392B', linewidth=2)
ax.add_patch(box5)
ax.text(9.9, 5.2, 'Experimental Validation', fontsize=11, fontweight='bold', ha='center', color='#C0392B')
ax.text(9.9, 4.8, '• Nearest neighbor lookup', fontsize=9, ha='center')
ax.text(9.9, 4.5, '• GP calibrated prediction', fontsize=9, ha='center')
ax.text(9.9, 4.2, '• Uncertainty quantification', fontsize=9, ha='center')
ax.text(9.9, 3.9, '• Select top candidates', fontsize=9, ha='center')

# Block 6: Output
box6 = FancyBboxPatch((5.3, 0.5), 3.2, 1.5, boxstyle="round,pad=0.1", 
                       facecolor='#FEF9E7', edgecolor='#D4AC0D', linewidth=2)
ax.add_patch(box6)
ax.text(6.9, 1.6, 'New Vitrimer Candidates', fontsize=11, fontweight='bold', ha='center', color='#D4AC0D')
ax.text(6.9, 1.2, '• Optimized acid-epoxide pairs', fontsize=9, ha='center')
ax.text(6.9, 0.9, '• Targeted Tg properties', fontsize=9, ha='center')

# Arrows
arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
# Data -> GP
ax.annotate('', xy=(5.3, 7.6), xytext=(3.5, 7.6), arrowprops=arrow_style)
# Data -> VAE
ax.annotate('', xy=(10.3, 7.6), xytext=(7.8, 7.6), arrowprops=arrow_style)
ax.text(9.0, 7.9, 'Calibrated Tg', fontsize=8, ha='center', style='italic')
# GP -> Inverse
ax.annotate('', xy=(4.9, 5.7), xytext=(6.9, 6.5), arrowprops=arrow_style)
# VAE -> Inverse
ax.annotate('', xy=(4.9, 5.3), xytext=(11.9, 6.5), arrowprops=arrow_style)
ax.text(8.5, 6.0, 'Latent space', fontsize=8, ha='center', style='italic')
# Inverse -> Validation
ax.annotate('', xy=(8.3, 4.6), xytext=(6.5, 4.6), arrowprops=arrow_style)
# Validation -> Output
ax.annotate('', xy=(6.9, 2.0), xytext=(9.9, 3.5), arrowprops=arrow_style)
ax.text(8.5, 2.6, 'Validated candidates', fontsize=8, ha='center', style='italic')
# Inverse -> Output
ax.annotate('', xy=(6.0, 2.0), xytext=(4.9, 3.5), arrowprops=arrow_style)

plt.tight_layout()
plt.savefig('report/images/fig00_framework.png', dpi=300, bbox_inches='tight')
plt.close()
print("Framework diagram saved.")
