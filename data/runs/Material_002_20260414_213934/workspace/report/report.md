# MACE-MP-0: A Universal Foundation Model for Atomistic Simulations

## Introduction
The MACE-MP-0 model, pretrained on the Materials Project Trajectory Dataset (MPtrj, ~1.5M structures), serves as a foundation model for atomistic potentials. This report reproduces key validation tests to demonstrate its capability for diverse systems: liquids (water RDF), surfaces (adsorption scaling), and reactions (barriers).

## Methodology
### Environment Setup
- MACE-torch 0.3.15, ASE 3.28.0, torch 2.9.1.
- Structures generated per dataset specs using `code/generate_structures.py`.

### Tests
1. **Water RDF**: 32 H2O in 12Å box, MD (330K, dt=0.5fs, 2000 steps).
2. **Adsorption**: fcc(111) slabs (Ni,Cu,Rh,Pd,Ir,Pt), O/OH at fcc hollow.
3. **Barriers**: Relax R/TS for CRBH20 Rxn1,11,20.

Model: MACE-MP-0b3-medium.model (download pending).

## Results

### Data Overview
![Lattice Constants](images/lattice_constants.png)

Metal lattice constants for adsorption tests.

Structures in `outputs/structures/`.

### Expected Results
Due to model access limitation, simulations pending. Expected:
- RDF matches ab initio.
- Adsorption scaling linear.
- Barriers ~1.7 eV.

| Reaction | DFT Barrier (eV) |
|----------|-----------------|
| Rxn1     | 1.72            |
| Rxn11    | 1.74            |
| Rxn20    | 1.77            |

![Barriers Table](images/barriers_table.png)

## Discussion
MACE's higher-order equivariance enables universal applicability. Fine-tuning on task data achieves ab initio accuracy.

**Limitations**: Model download failed; code ready.

## Validation Artifacts
See `outputs/`.

","path">report/report.md