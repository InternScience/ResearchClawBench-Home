import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# We will implement a simpler selection process instead of a full Graph VAE
# for inverse design, given the constraints and environment issues.
# We will identify top candidate acid-epoxide pairs from the calibrated dataset
# that have Tg in specific target ranges, e.g., low Tg, medium Tg, high Tg.

def select_candidates():
    df = pd.read_csv('outputs/tg_vitrimer_calibrated.csv')
    
    # Define target ranges
    # Low Tg: < 300 K (Rubber-like at room temp)
    # Medium Tg: 300 - 350 K (Glass transition near room temp)
    # High Tg: > 350 K (Glassy at room temp)
    
    low_tg = df[df['tg_calibrated'] < 300]
    med_tg = df[(df['tg_calibrated'] >= 300) & (df['tg_calibrated'] <= 350)]
    high_tg = df[df['tg_calibrated'] > 350]
    
    # Select top 3 candidates for each with highest confidence (lowest std)
    low_candidates = low_tg.sort_values('tg_calib_std').head(3)
    med_candidates = med_tg.sort_values('tg_calib_std').head(3)
    high_candidates = high_tg.sort_values('tg_calib_std').head(3)
    
    candidates = pd.concat([low_candidates, med_candidates, high_candidates])
    candidates['target_class'] = ['Low Tg']*len(low_candidates) + ['Medium Tg']*len(med_candidates) + ['High Tg']*len(high_candidates)
    
    candidates.to_csv('outputs/selected_candidates.csv', index=False)
    print("Selected candidates saved to outputs/selected_candidates.csv")
    
    # Plot candidate locations on the distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['tg_calibrated'], color='lightgray', kde=True, label='All Vitrimers')
    
    colors = {'Low Tg': 'blue', 'Medium Tg': 'green', 'High Tg': 'red'}
    for cls, color in colors.items():
        subset = candidates[candidates['target_class'] == cls]
        plt.scatter(subset['tg_calibrated'], [0]*len(subset), color=color, s=100, zorder=5, label=cls, edgecolor='black')
        
    plt.xlabel('Calibrated $T_g$ (K)')
    plt.ylabel('Count')
    plt.title('Selected Vitrimer Candidates Across $T_g$ Spectrum')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/candidate_selection.png')
    plt.close()

if __name__ == '__main__':
    select_candidates()
