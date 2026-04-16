# Research Plan: Multi-component Icosahedral Nanoclusters

## Objective
Establish a universal theoretical framework for the rational design of multi-component nanoclusters and nanoparticles with specific symmetry (chiral or achiral) and compositional sequences, predict their stability and growth behavior.

## Deliverables
1. Predicted stable multi-shell icosahedral structures (e.g., $\mathrm{Na_{13}@K_{32}}$, $\mathrm{Ni_{147}@Ag_{192}}$).
2. Optimal size mismatch values between adjacent shells.
3. Shell sequences and paths formed via self-assembly in growth simulations.
4. `report/report.md` with methodology, results, discussion, and figures.

## Data Available
- `data/Multi-component Icosahedral Reproduction Data.txt`: Contains parameters and result data for reproducing all simulation experiments.
  - Core Theory Data (hexagonal coords, Mackay sequence, new sequence b=5, chiral labels, geometric constants, shell colors)
  - Experimental Verification Data (atomic radii, pair compatibility, optimal mismatch ranges, multi-component clusters, shell energies, mismatch params, experimental points)
  - Dynamic Growth Simulation Data (growth parameters, path probability weights, initial seeds, deposition sequences, growth results, path selection stats, lj parameters, thermodynamic params)

## Methodology
1. **Core Theory**: Define the geometric framework based on Archimedean lattices (from Twarock & Luque paper and data) to predict magic numbers and shell sequences for multi-component icosahedra.
2. **Size Mismatch**: Calculate the theoretical optimal size mismatch between shells and compare with experimental/simulation data.
3. **Stability & Energy**: Analyze shell energies for different chiral/achiral configurations.
4. **Growth Simulation**: Analyze dynamic growth simulation data, path selection, and mismatch-driven self-assembly.

## Implementation Steps
1. Parse the data file into Python data structures.
2. Generate Figure 1: Core Theory - Magic numbers and shell sequences (Mackay vs. new sequence b=5).
3. Generate Figure 2: Size Mismatch & Stability - Optimal mismatch ranges for different chiral categories and shell energies.
4. Generate Figure 3: Experimental Verification - Measured vs. theoretical size mismatch.
5. Generate Figure 4: Growth Dynamics - Average mismatch over time for different chiral categories and path selection statistics.
6. Draft the report `report/report.md` incorporating the figures and analysis.
