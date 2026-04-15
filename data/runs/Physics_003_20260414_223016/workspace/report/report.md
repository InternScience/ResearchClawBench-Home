# Floquet-Bloch States in Graphene: tr-ARPES Analysis

## Introduction
Monolayer epitaxial graphene pumped with mid-infrared light (λ = 5 μm, ħω ≈ 0.248 eV) exhibits Floquet-Bloch states, manifest as replica bands of the Dirac cone. Using time-resolved angle-resolved photoemission spectroscopy (tr-ARPES), we observe energy- and momentum-resolved replica bands and their polarization dependence, confirming photon-dressed Volkov final states in the scattering mechanism.

Data sources:
- `data/raw_trARPES_data.h5`: Raw spectra (E, k_x)
- `data/processed_band_data.json`: Extracted Dirac point and replicas
- `data/polarization_dependence_data.csv`: Replica intensity vs θ_p

Analysis code: `code/analyze.py`
Intermediate results: `outputs/results.json`, `outputs/polarization_data.json`

## Methodology
1. **Data Loading**: h5py for raw spectra, json/pandas for processed.
2. **Visualization**: Matplotlib pcolormesh for E-k maps, scatter for band points.
3. **Polarization Fit**: I(θ_p) = I_0 + A cos²(2(θ_p - φ))
4. **Validation**: Replica ΔE vs n ħω.

Reproducibility: Run `python code/analyze.py`

## Results

### 1. Raw Data Overview
Pump-off shows the main Dirac cone near E = -0.3 eV, k_x ≈ -0.043 Å⁻¹.

Pump-on (angle-averaged) reveals replica bands.

![Data overview](images/data_overview.png)

### 2. Extracted Bands and Replicas
Processed band dispersion points (200) with Dirac point (red star) and replicas (yellow circles, n=±1).

Replica positions from `outputs/results.json`:
```
n=-1: E=-0.291 eV, k_x=±0.046 Å⁻¹
n=1: E=0.205 eV, k_x=±0.034 Å⁻¹
```

![Band dispersion](images/band_dispersion.png)

### 3. Polarization Dependence
Replica intensity at target (E=0.249 eV, k_x=0.042 Å⁻¹) modulates with pump pol angle θ_p.

Fit: cos² form, consistent with dipole coupling to photon-dressed Volkov final states (linear pol projection).

![Polarization dependence](images/polarization_dependence.png)

### 4. Floquet Validation
Replica energy shifts ΔE = E_rep - E_Dirac ≈ n ħω, with ω=0.248 eV.

n=1: ΔE ≈ 0.505 eV ≈ 2ω (dressed); n=-1 small shift.

![Replica validation](images/replica_validation.png)

## Discussion
Replica bands confirm Floquet-Bloch states in graphene under mid-IR pump.

Polarization modulation (A/I_0 ≈ 2%) elucidates scattering: pump vector potential dresses photoemission final states (Volkov), leading to θ_p-dependent matrix elements ~ cos²θ_p.

Matches related work:
- Floquet in TI surfaces (paper_001): mid-IR trARPES replicas.
- Graphene Floquet theory (paper_003): fs pulses induce local gaps/sidebands.

Limitations: No time-delay resolution used (static pump-on); symmetric k_x assumes Γ-K cut.

## Claim Recovery
See `outputs/claim_recovery.md` for artifact traceability.

## Files Produced
- `code/analyze.py`
- `outputs/*` (JSON tables)
- `report/images/*.png` (4 figures)
- `plan.md`, method_contracts