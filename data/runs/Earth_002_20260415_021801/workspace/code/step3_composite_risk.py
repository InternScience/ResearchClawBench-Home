"""
Step 3: Compute composite risk index combining SLR and TC risk.
Also project future TC regime shifts under warming scenarios.
"""
import numpy as np
import pandas as pd
import json
import os

os.makedirs('outputs', exist_ok=True)

# Load combined data
df = pd.read_csv('outputs/mangrove_tc_metrics.csv')
print(f"Loaded {len(df)} mangrove points")

# ============================================================
# SLR Risk Component
# ============================================================
# Based on Saintilan et al. (2023):
# - <4 mm/yr: Low risk (mangroves can adjust)
# - 4-7 mm/yr: Moderate risk (deficit likely)
# - 7-10 mm/yr: High risk (deficit highly likely)
# - >=10 mm/yr: Very High risk (severe deficit)

def slr_risk_score(rate):
    """Convert SLR rate to normalized risk score [0,1] using piecewise linear."""
    if rate < 4:
        return rate / 4 * 0.25  # 0-0.25 for Low
    elif rate < 7:
        return 0.25 + (rate - 4) / 3 * 0.25  # 0.25-0.5 for Moderate
    elif rate < 10:
        return 0.5 + (rate - 7) / 3 * 0.25  # 0.5-0.75 for High
    else:
        return min(0.75 + (rate - 10) / 10 * 0.25, 1.0)  # 0.75-1.0 for Very High

def slr_risk_category(score):
    if score < 0.25:
        return 'Low'
    elif score < 0.5:
        return 'Moderate'
    elif score < 0.75:
        return 'High'
    else:
        return 'Very High'

# ============================================================
# TC Risk Component
# ============================================================
# Based on Mo et al. (2023) and Kropf et al. (2023):
# - Major TCs (Cat 3+, wind >= 50 m/s) cause substantial damage
# - Intense TCs (Cat 4+, wind >= 59 m/s) contribute 97% of risk
# - TC frequency and intensity both matter
# - Recovery time ~2 years for Cat 5 damage
# 
# TC risk = f(frequency of major TCs, max wind speed)
# Higher frequency = less recovery time = higher risk
# Higher intensity = more damage = higher risk

# Future TC projections (Knutson et al. 2020, Mo et al. 2023):
# - Global TC frequency: ~-10% to -20% 
# - Major TC (Cat 3-5) frequency: +10% to +20%
# - Cat 4-5 frequency: +20% to +40%
# - Peak wind intensity: +5% to +10%
# Regional variation is significant

tc_future_factors = {
    'SSP2-4.5': {
        'major_freq_change': 1.10,   # +10% major TC frequency
        'intense_freq_change': 1.15,  # +15% intense TC frequency
        'wind_intensity_change': 1.05, # +5% wind intensity
    },
    'SSP3-7.0': {
        'major_freq_change': 1.20,
        'intense_freq_change': 1.30,
        'wind_intensity_change': 1.08,
    },
    'SSP5-8.5': {
        'major_freq_change': 1.30,
        'intense_freq_change': 1.40,
        'wind_intensity_change': 1.10,
    },
}

def tc_risk_score(freq_major, freq_intense, max_wind, future_factor=None):
    """
    Compute TC risk score [0,1].
    Combines frequency of major TCs and maximum wind intensity.
    """
    if future_factor:
        freq_major *= future_factor['major_freq_change']
        freq_intense *= future_factor['intense_freq_change']
        max_wind *= future_factor['wind_intensity_change']
    
    # Frequency component: based on major TC frequency
    # 0/yr = no risk, 0.5/yr = high, 1.0/yr = very high
    freq_score = min(freq_major / 0.5, 1.0)
    
    # Intensity component: based on max wind
    # <33 m/s = no TC risk, 50 m/s = Cat 3, 70 m/s = Cat 5
    if max_wind < 33:
        wind_score = 0.0
    elif max_wind < 50:
        wind_score = (max_wind - 33) / 17 * 0.3  # 0-0.3
    elif max_wind < 70:
        wind_score = 0.3 + (max_wind - 50) / 20 * 0.4  # 0.3-0.7
    else:
        wind_score = min(0.7 + (max_wind - 70) / 50 * 0.3, 1.0)  # 0.7-1.0
    
    # Combined: weighted average (frequency 40%, intensity 60%)
    # If no TCs at all, score is 0
    if freq_major == 0 and max_wind < 33:
        return 0.0
    
    return 0.4 * freq_score + 0.6 * wind_score

def tc_risk_category(score):
    if score < 0.15:
        return 'Low'
    elif score < 0.4:
        return 'Moderate'
    elif score < 0.65:
        return 'High'
    else:
        return 'Very High'

# ============================================================
# Composite Risk Index
# ============================================================
# Equal weights for SLR and TC (both are critical threats)
# CRI = 0.5 * SLR_risk + 0.5 * TC_risk

def composite_risk(slr_score, tc_score):
    return 0.5 * slr_score + 0.5 * tc_score

def composite_risk_category(score):
    if score < 0.25:
        return 'Low'
    elif score < 0.5:
        return 'Moderate'
    elif score < 0.75:
        return 'High'
    else:
        return 'Very High'

# ============================================================
# Calculate for each scenario
# ============================================================
scenarios = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

results = {}

for scenario in scenarios:
    print(f"\n{'='*60}")
    print(f"Processing {scenario}...")
    
    slr_col = f'slr_rate_{scenario}'
    slr_rates = df[slr_col].values
    
    # SLR risk scores
    slr_scores = np.array([slr_risk_score(r) for r in slr_rates])
    slr_cats = np.array([slr_risk_category(s) for s in slr_scores])
    
    # TC risk scores (with future projection)
    future_factor = tc_future_factors[scenario]
    tc_scores = np.array([
        tc_risk_score(
            df['tc_freq_major_tc'].values[i],
            df['tc_freq_intense_tc'].values[i],
            df['tc_maxwind_all_tc'].values[i],
            future_factor
        )
        for i in range(len(df))
    ])
    tc_cats = np.array([tc_risk_category(s) for s in tc_scores])
    
    # Composite risk
    cri_scores = np.array([composite_risk(slr_scores[i], tc_scores[i]) for i in range(len(df))])
    cri_cats = np.array([composite_risk_category(s) for s in cri_scores])
    
    # Store in dataframe
    df[f'slr_risk_{scenario}'] = slr_scores
    df[f'slr_cat_{scenario}'] = slr_cats
    df[f'tc_risk_{scenario}'] = tc_scores
    df[f'tc_cat_{scenario}'] = tc_cats
    df[f'cri_{scenario}'] = cri_scores
    df[f'cri_cat_{scenario}'] = cri_cats
    
    # Summary
    cat_counts = pd.Series(cri_cats).value_counts()
    total = len(cri_cats)
    
    scenario_results = {
        'slr_mean_rate': float(np.mean(slr_rates)),
        'slr_risk_mean': float(np.mean(slr_scores)),
        'tc_risk_mean': float(np.mean(tc_scores)),
        'cri_mean': float(np.mean(cri_scores)),
        'cri_categories': {
            'Low': int(cat_counts.get('Low', 0)),
            'Moderate': int(cat_counts.get('Moderate', 0)),
            'High': int(cat_counts.get('High', 0)),
            'Very High': int(cat_counts.get('Very High', 0)),
        },
        'pct_very_high': float(cat_counts.get('Very High', 0) / total * 100),
        'pct_high_or_very_high': float((cat_counts.get('High', 0) + cat_counts.get('Very High', 0)) / total * 100),
    }
    results[scenario] = scenario_results
    
    print(f"  SLR risk (mean): {scenario_results['slr_risk_mean']:.3f}")
    print(f"  TC risk (mean): {scenario_results['tc_risk_mean']:.3f}")
    print(f"  CRI (mean): {scenario_results['cri_mean']:.3f}")
    print(f"  CRI categories: {scenario_results['cri_categories']}")
    print(f"  % High or Very High: {scenario_results['pct_high_or_very_high']:.1f}%")

# Save full results
df.to_csv('outputs/mangrove_composite_risk.csv', index=False)
print("\nSaved composite risk to outputs/mangrove_composite_risk.csv")

with open('outputs/composite_risk_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved summary to outputs/composite_risk_summary.json")
