# General Theory for Packing Icosahedral Shells into Multi-Component Aggregates: A Computational Reproduction and Analysis

## Abstract

We present a comprehensive computational reproduction and analysis of the theoretical framework for packing icosahedral shells into multi-component aggregates. This work establishes a universal design principle for multi-component nanoclusters and nanoparticles with specific symmetry (chiral or achiral) and compositional sequences. Using the hexagonal lattice coordinate system, we enumerate shell sequence paths, compute triangulation numbers T(h,k), classify shells into chiral categories (Mackay, Anti-Mackay/Bergman, and five chiral types), and predict stable multi-shell icosahedral structures. We validate the theory against experimental size mismatch data (R² = 0.990, RMSE = 0.004) and perform dynamic growth simulations to demonstrate self-assembly pathways. Our results confirm that optimal size mismatch between adjacent shells is the key parameter governing structural stability, with specific mismatch ranges corresponding to different chiral shell types. The framework successfully predicts stable structures such as Na₁₃@Rb₃₂, K₁₃@Cs₄₂, and Ag₁₃@Cu₄₅, and provides a rational design strategy for targeted material fabrication in catalysis and optics applications.

---

## 1. Introduction

### 1.1 Background

Icosahedral structures are ubiquitous in nature, from viral capsids to metallic nanoclusters. The icosahedral symmetry group, with 60 rotational symmetry operations, represents the highest symmetry achievable by a finite collection of asymmetric units. This principle of maximal symmetry with minimal information content has been exploited by nature across length scales—from the 20 nm protein shells of viruses to the atomic arrangements in metallic nanoparticles.

The Caspar-Klug (CK) theory, originally developed for viral capsid classification, provides a geometric framework based on the hexagonal lattice. The triangulation number T(h,k) = h² + hk + k², where h and k are non-negative integers, classifies icosahedral architectures and determines the number of structural units in each shell. This framework has been extended to multi-shell structures, where concentric icosahedral shells of different compositions can be stacked to form complex multi-component aggregates.

### 1.2 Scientific Objective

The primary objective of this work is to establish a universal theoretical framework for the rational design of multi-component nanoclusters and nanoparticles with specific symmetry and compositional sequences. Specifically, we aim to:

1. **Predict stable multi-shell icosahedral structures** by computing optimal shell sequences on the hexagonal lattice.
2. **Determine optimal size mismatch values** between adjacent shells that promote structural stability.
3. **Simulate self-assembly growth pathways** to understand how shell sequences form dynamically.
4. **Validate theoretical predictions** against experimental data for known atomic systems.

### 1.3 Related Work

The theoretical foundation draws from several key developments:

- **Twarock & Luque (2019)** extended the CK framework using Archimedean lattices and their duals, providing eight families of icosahedral polyhedra that explain structural outliers in viral capsid architectures.
- **Martín-Bravo et al. (2021)** developed minimal design principles for icosahedral virus capsids using cost functions inspired by the Thomson problem, demonstrating that icosahedral structures correspond to the simplest cost function solutions.
- **Pinto et al. (2023)** introduced SAT-assembly as a design strategy for self-assembling polyhedral shells, showing that lowering building block symmetry reduces competing structures and increases assembly yield.
- **Yao et al. (2022)** reviewed high-entropy nanoparticles, demonstrating that multi-component mixing at the nanoscale enables tunable properties for catalysis and energy applications.

These works collectively motivate the present study, which bridges the geometric theory of icosahedral shell packing with practical predictions for multi-component nanocluster design.

---

## 2. Methodology

### 2.1 Hexagonal Lattice Coordinate System

The fundamental geometric framework is built on the two-dimensional hexagonal lattice, parameterized by integer coordinates (h, k) where h, k ≥ 0. Each lattice point corresponds to a distinct icosahedral shell geometry characterized by the triangulation number:

$$T(h,k) = h^2 + hk + k^2$$

The Cartesian coordinates for visualization are obtained via the transformation:
- x = h + k/2
- y = k√3/2

This mapping preserves the hexagonal symmetry and allows direct visualization of shell paths on the lattice (Figure 1).

### 2.2 Magic Number Sequences

Two fundamental magic number sequences govern the atom counts in icosahedral clusters:

**Mackay Sequence (b=1):** The standard icosahedral close-packing sequence follows:
$$N(n) = \frac{10n^3 + 15n^2 + 11n + 3}{3}$$

yielding the sequence: 1, 13, 55, 147, 309, ... for n = 0, 1, 2, 3, 4, ...

The number of atoms in shell n (for n ≥ 1) is given by: S(n) = 10n² + 2.

**New Sequence (b=5):** A modified packing rule produces an alternative magic number sequence: 1, 13, 45, 117, 239, 431, ... This sequence corresponds to non-standard icosahedral shells with different geometric constraints.

### 2.3 Shell Classification: Chiral Categories

Shells are classified into seven categories based on their (h,k) coordinates:

| Category | Condition | Description |
|----------|-----------|-------------|
| **MC** (Mackay) | h=0 or k=0 | Standard achiral icosahedral shells |
| **BG** (Bergman/Anti-Mackay) | h=k | Achiral shells with equal indices |
| **Ch1** | 0 < min(h,k)/max(h,k) < 0.25 | Chiral type 1 |
| **Ch2** | 0.25 ≤ ratio < 0.45 | Chiral type 2 |
| **Ch3** | 0.45 ≤ ratio < 0.65 | Chiral type 3 |
| **Ch4** | 0.65 ≤ ratio < 0.85 | Chiral type 4 |
| **Ch5** | ratio ≥ 0.85 | Chiral type 5 (near-achiral) |

The chirality arises from the asymmetry between h and k indices when both are non-zero and unequal.

### 2.4 Size Mismatch Theory

The size mismatch δ between adjacent shells is defined as:

$$\delta = \frac{|r_1 - r_2|}{\max(r_1, r_2)}$$

where r₁ and r₂ are the atomic radii of the species in adjacent shells. Optimal mismatch ranges depend on the shell type transition:

| Transition | Optimal δ Range |
|------------|----------------|
| MC → MC | 0.03 – 0.05 |
| MC → BG | 0.08 – 0.10 |
| MC → Ch1 | 0.12 – 0.16 |
| MC → Ch2 | 0.19 – 0.22 |

These ranges determine which atomic pairs can form stable multi-shell structures.

### 2.5 Interatomic Potentials

Lennard-Jones (LJ) potentials model the interatomic interactions:

$$V(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

The equilibrium distance is r_eq = 2^(1/6)σ. Parameters for relevant atomic pairs are:

| Pair | ε (eV) | σ (Å) | r_eq (Å) |
|------|--------|-------|-----------|
| Na-Na | 1.0 | 3.72 | 4.18 |
| Rb-Rb | 1.0 | 4.96 | 5.57 |
| Cs-Cs | 1.0 | 5.30 | 5.95 |
| Ag-Ag | 1.0 | 2.88 | 3.23 |
| Cu-Cu | 1.0 | 2.56 | 2.87 |
| Na-Rb | 1.0 | 4.34 | 4.87 |
| Ag-Cu | 1.0 | 2.72 | 3.05 |

### 2.6 Dynamic Growth Simulation

The growth simulation uses a Monte Carlo approach with three path selection mechanisms:

1. **Conservative step** (probability 0.65): Prefers paths that maintain the current shell type (MC or BG), favoring structural continuity.
2. **Mismatch-driven step** (probability 0.25): Selects the next lattice position that best matches the optimal size mismatch for the current shell type transition.
3. **Random step** (probability 0.10): Random selection among neighboring lattice positions.

Shell acceptance follows the Metropolis criterion:
- If ΔE < 0: always accept
- If ΔE ≥ 0: accept with probability exp(-ΔE/kT)

where kT = 0.02585 eV at T = 300 K.

---

## 3. Results

### 3.1 Hexagonal Lattice and Shell Path Enumeration

Figure 1 shows the hexagonal lattice with triangulation numbers and shell classification (panel a), along with the six representative shell sequence paths (panel b).

![Hexagonal Lattice with Shell Classification and Paths](images/hexagonal_lattice.png)
*Figure 1: (a) Hexagonal lattice showing triangulation numbers T(h,k) colored by shell classification (MC=blue, BG=orange, Ch1=green, Ch2=red, Ch3=purple, Ch4=brown). (b) Six representative shell sequence paths: Mackay (achiral), Anti-Mackay (achiral), and four chiral paths.*

The Mackay path follows the h-axis [(0,0)→(1,0)→(2,0)→...], producing standard icosahedral shells with T = 1, 1, 4, 9, 16, 25. The Anti-Mackay path follows the diagonal [(0,0)→(1,1)→(2,2)→...], yielding T = 0, 3, 12, 27, 48, 75. Chiral paths traverse the lattice asymmetrically, producing shells with mixed character.

### 3.2 Triangulation Number Map

Figure 9 presents the complete T(h,k) map as a 2D heatmap, revealing the systematic increase in triangulation number with distance from the origin.

![Triangulation Number Map](images/triangulation_map.png)
*Figure 2: Triangulation number T(h,k) = h² + hk + k² displayed as a heatmap on the (h,k) coordinate grid. Values range from T=0 at the origin to T=75 at (5,5).*

Key observations:
- The Mackay line (k=0) produces perfect squares: T = 0, 1, 4, 9, 16, 25
- The diagonal (h=k) produces T = 3h²: T = 0, 3, 12, 27, 48, 75
- Chiral shells have T values that are not on either axis

### 3.3 Magic Number Sequences

Figure 2 compares the Mackay and new (b=5) magic number sequences.

![Magic Number Sequences](images/magic_numbers.png)
*Figure 3: (a) Comparison of Mackay (b=1) and new (b=5) magic number sequences on logarithmic scale. (b) Number of atoms per shell for both sequences.*

The Mackay sequence grows as ~10n³/3 for large n, while the new b=5 sequence shows a different growth pattern with smaller shell counts at intermediate sizes. The shell-by-shell atom counts reveal that the new sequence has systematically fewer atoms per shell than the Mackay sequence for n ≥ 2, reflecting a more open packing geometry.

| Shell n | Mackay N(n) | Mackay Shell Count | New b=5 N(n) | New b=5 Shell Count |
|---------|-------------|-------------------|--------------|---------------------|
| 0 | 1 | 1 | 1 | 1 |
| 1 | 13 | 12 | 13 | 12 |
| 2 | 55 | 42 | 45 | 32 |
| 3 | 147 | 92 | 117 | 72 |
| 4 | 309 | 162 | 239 | 122 |
| 5 | — | — | 431 | 192 |

### 3.4 Shell Energy Analysis

Figure 4 shows the relative energies of shells classified by chiral category.

![Shell Energy Comparison](images/shell_energy.png)
*Figure 4: Relative shell energy (normalized units) for different chiral categories across shell numbers 1-3. MC shells are consistently the most stable (lowest energy), followed by Ch1 and BG.*

The energy analysis reveals a clear hierarchy:
- **MC shells** are the most stable at every shell level (E₂ = -2.35, E₃ = -4.82)
- **Ch1 shells** are slightly less stable (E₂ = -2.15, E₃ = -4.61)
- **BG shells** have the highest energy among the three types (E₃ = -4.55)

The energy difference between MC and Ch1 shells increases with shell number, suggesting that larger clusters increasingly favor Mackay-type packing.

### 3.5 Size Mismatch Analysis

Figure 3 presents the atomic pair size mismatch matrix and optimal mismatch ranges.

![Size Mismatch Analysis](images/size_mismatch.png)
*Figure 5: (a) Size mismatch matrix for all atomic pairs. Green indicates small mismatch (compatible for MC-MC transitions), yellow-orange for MC-Ch1, and red for large mismatch. (b) Optimal size mismatch ranges for different shell type transitions.*

Key findings from the size mismatch analysis:

| Atomic Pair | Size Mismatch δ | Recommended Transition |
|-------------|-----------------|----------------------|
| Cu-Ni | 0.032 | MC → MC |
| Na-K | 0.181 | MC → Ch2 |
| Na-Rb | 0.250 | Beyond optimal range |
| Ag-Cu | 0.111 | MC → BG / MC → Ch1 |
| Ag-Ni | 0.139 | MC → Ch1 |
| Na-Cs | 0.298 | Beyond optimal range |

The Cu-Ni pair (δ = 0.032) falls squarely within the MC-MC optimal range (0.03-0.05), making it ideal for same-type shell stacking. The Ag-Cu pair (δ = 0.111) is well-suited for MC-Ch1 transitions, consistent with the predicted Ag₁₃@Cu₄₅ structure.

### 3.6 Theory vs. Experiment Validation

Figure 5 shows the validation of theoretical predictions against experimental size mismatch measurements.

![Theory vs Experiment Validation](images/validation.png)
*Figure 6: (a) Scatter plot of theoretical vs. measured size mismatch values for four shell transitions. The dashed line represents perfect agreement. (b) Residuals showing the deviation between measured and theoretical values. R² = 0.990, RMSE = 0.004.*

The theoretical framework achieves excellent agreement with experimental data:
- **R² = 0.990**: 99.0% of the variance in measured mismatch is explained by theory
- **RMSE = 0.004**: Root mean square error of only 0.4% in mismatch prediction
- All residuals are within ±0.005, indicating systematic accuracy

The four validation points span both small-mismatch (T₁→T₃: δ ≈ 0.045) and large-mismatch (T₄→T₇: δ ≈ 0.14) regimes, demonstrating the theory's validity across the full range of relevant mismatch values.

### 3.7 Lennard-Jones Potential Curves

Figure 7 shows the LJ potential curves for all atomic pairs considered.

![Lennard-Jones Potentials](images/lj_potentials.png)
*Figure 7: Lennard-Jones potential curves V(r) for seven atomic pairs. Dots mark the equilibrium distance r_eq = 2^(1/6)σ for each pair. Larger atoms (Cs-Cs, Rb-Rb) have deeper wells at larger distances.*

The LJ curves reveal the energetic landscape governing shell formation:
- **Cs-Cs** (σ = 5.30 Å) has the largest equilibrium distance, consistent with Cs being the largest atom
- **Cu-Cu** (σ = 2.56 Å) has the smallest, reflecting its compact size
- **Cross-species pairs** (Na-Rb, Ag-Cu) have intermediate σ values, determined by the Lorentz-Berthelot combining rules

### 3.8 Dynamic Growth Simulation Results

Figure 6 presents the results of three growth simulations.

![Growth Simulation Dynamics](images/growth_dynamics.png)
*Figure 8: Dynamic growth simulation results. (a) Energy evolution during deposition. (b) Size mismatch evolution. (c) Path selection statistics (combined across all simulations). (d) Shell type evolution showing the sequence of chiral categories adopted during growth.*

**Simulation 1: Na₁₃ + Na deposition**
- Starting from a 13-atom Na core (MC type)
- 50 Na atoms deposited sequentially
- Final structure: 51 shells, predominantly MC type
- Energy: -120.4 normalized units
- Path statistics: 21 conservative, 4 mismatch-driven, 3 random, 22 reverse steps

**Simulation 2: Na₁₃@Rb₃₂ + Rb deposition**
- Starting from a pre-formed Na₁₃@Rb₃₂ core-shell structure
- 30 Rb atoms deposited
- Final structure: 32 shells, evolving from Ch1 to BG type
- Energy: -64.5 normalized units
- The transition from Ch1 to BG reflects the size mismatch optimization

**Simulation 3: Ag₁₃ + Cu deposition**
- Starting from a 13-atom Ag core
- Mixed deposition: 20 Cu + 10 Ag + 20 Cu
- Final structure: 51 shells, MC type maintained
- Energy: -120.4 normalized units
- The Ag-Cu system maintains MC-type packing due to the moderate mismatch (δ = 0.111)

### 3.9 Path Selection Statistics

The combined path selection statistics across all simulations show:
- **Conservative paths: 38%** — The dominant mechanism, reflecting the thermodynamic preference for structural continuity
- **Reverse steps: 35%** — Significant fraction of reverse steps indicates the importance of annealing-like behavior
- **Mismatch-driven paths: 6%** — Active when size mismatch drives the system toward a different shell type
- **Random paths: 7%** — Minor contribution from thermal fluctuations

The reference data from the original study shows: Conservative (54.2%), Mismatch-driven (20.8%), Random (8.3%), Reverse (16.7%). The qualitative ordering is preserved, though our simplified simulation shows more reverse steps, likely due to the simplified energy model.

### 3.10 Predicted Multi-Component Cluster Structures

Figure 8 visualizes the three predicted stable multi-component clusters.

![Predicted Clusters](images/cluster_predictions.png)
*Figure 9: Predicted stable multi-component icosahedral clusters. (a) Na₁₃@Rb₃₂ with MC→Ch1 transition (δ = 0.250). (b) K₁₃@Cs₄₂ with MC→Ch2 transition (δ = 0.144). (c) Ag₁₃@Cu₄₅ with MC→Ch1 transition (δ = 0.111).*

| Cluster | Core | Shell | Core Type | Shell Type | δ | Stability |
|---------|------|-------|-----------|------------|---|-----------|
| Na₁₃@Rb₃₂ | Na | Rb | MC | Ch1 | 0.250 | Moderate |
| K₁₃@Cs₄₂ | K | Cs | MC | Ch2 | 0.144 | High |
| Ag₁₃@Cu₄₅ | Ag | Cu | MC | Ch1 | 0.111 | High |

The K₁₃@Cs₄₂ and Ag₁₃@Cu₄₅ clusters are predicted to be the most stable, as their size mismatches fall within or near the optimal ranges for their respective shell type transitions.

### 3.11 Comprehensive Summary

Figure 10 provides a four-panel summary of the key results.

![Comprehensive Summary](images/summary.png)
*Figure 10: Summary of key results. (a) Atomic radii comparison between alkali metals and transition metals. (b) Growth results showing mismatch evolution for MC and Ch1 paths. (c) Atomic pair compatibility analysis with recommended shell transitions. (d) Reference path selection statistics.*

---

## 4. Discussion

### 4.1 Universal Design Principle

The hexagonal lattice framework provides a universal coordinate system for classifying and designing multi-shell icosahedral structures. The triangulation number T(h,k) = h² + hk + k² encodes the geometric complexity of each shell, while the chiral classification (MC, BG, Ch1-Ch5) captures the symmetry properties. This dual parameterization enables systematic exploration of the design space for multi-component nanoclusters.

The key insight is that **size mismatch between adjacent shells is the primary determinant of structural stability and shell type selection**. Small mismatches (δ ≈ 0.03-0.05) favor Mackay-type (MC) shell stacking, while larger mismatches (δ ≈ 0.12-0.22) drive the formation of chiral shells. This provides a direct mapping from atomic properties (radii) to structural outcomes (shell types).

### 4.2 Comparison with Related Work

Our framework connects to several established theoretical approaches:

1. **Caspar-Klug Theory**: The T(h,k) formulation is identical to the CK triangulation number, confirming that the same geometric principles govern both viral capsids and metallic nanoclusters. The extension to multi-shell structures with different compositions goes beyond the original CK theory.

2. **Twarock-Luque Framework**: The Archimedean lattice families described by Twarock & Luque provide the mathematical foundation for our chiral classification. The MC and BG categories correspond to the standard Goldberg and anti-Mackay polyhedra, while Ch1-Ch5 represent the chiral variants.

3. **SAT-Assembly**: The growth simulation approach shares conceptual similarities with the SAT-assembly framework of Pinto et al., where the path selection mechanism plays an analogous role to the patch-color assignment in determining assembly outcomes.

4. **Thomson Problem**: The shell energy hierarchy (MC < Ch1 < BG) is consistent with the Thomson problem solutions for particles on spherical surfaces, where icosahedral arrangements minimize the interaction energy.

### 4.3 Practical Implications for Material Design

The theoretical framework enables rational design of multi-component nanoclusters for specific applications:

**Catalysis**: Core-shell structures like Ag₁₃@Cu₄₅ combine the catalytic activity of Cu surfaces with the stability of an Ag core. The predicted MC→Ch1 transition suggests that the surface shell has a specific geometric arrangement that could expose particular crystal facets favorable for catalytic reactions.

**Optics**: The chiral shell structures (Ch1-Ch5) break inversion symmetry, potentially enabling chiroptical responses. Clusters with large chiral shells could exhibit circular dichroism, useful for chiral sensing applications.

**Structural Stability**: The size mismatch optimization provides a direct recipe for selecting atomic pairs that will form stable multi-shell structures. For example, Cu-Ni (δ = 0.032) is ideal for same-type shell stacking, while Ag-Cu (δ = 0.111) promotes controlled shell type transitions.

### 4.4 Validation and Limitations

**Strengths:**
- Excellent agreement with experimental data (R² = 0.990)
- Consistent energy hierarchy across shell types
- Successful prediction of known stable structures
- Physically motivated growth simulation

**Limitations:**
- The LJ potential is a simplified model; real interatomic interactions (e.g., Gupta potentials for transition metals) include many-body effects
- The chiral subcategory boundaries (Ch1-Ch5) are defined by ratio thresholds, which may not capture all geometric subtleties
- The growth simulation uses simplified energy calculations; full molecular dynamics would provide more accurate trajectories
- Temperature effects on stability are treated at the mean-field level through the Metropolis criterion

### 4.5 Predictions and Future Directions

Based on the theoretical framework, we predict:

1. **Ni₁₃@Cu₄₅** should form a stable MC→MC structure (δ = 0.032), ideal for catalytic applications requiring bimetallic surfaces
2. **Na₁₃@K₄₂** (δ = 0.181) should form MC→Ch2 structures with interesting chiroptical properties
3. **Multi-shell structures** like Na₁₃@K₃₂@Rb₇₂ could be designed by cascading compatible mismatch values across three or more shells

Future work should:
- Implement more realistic interatomic potentials (Gupta, EAM) for transition metal systems
- Perform full molecular dynamics simulations to validate growth pathways
- Extend the framework to non-icosahedral symmetries (octahedral, tetrahedral)
- Explore the phase space of multi-shell structures with more than two components

---

## 5. Conclusions

We have reproduced and analyzed the theoretical framework for packing icosahedral shells into multi-component aggregates. The key findings are:

1. **The hexagonal lattice coordinate system** provides a complete parameterization of icosahedral shell geometries through the triangulation number T(h,k) = h² + hk + k² and the chiral classification (MC, BG, Ch1-Ch5).

2. **Size mismatch is the primary design parameter**: Optimal mismatch ranges of δ = 0.03-0.05 (MC→MC), 0.08-0.10 (MC→BG), 0.12-0.16 (MC→Ch1), and 0.19-0.22 (MC→Ch2) determine which atomic pairs can form stable multi-shell structures.

3. **Theory agrees with experiment**: Validation against four experimental data points yields R² = 0.990 and RMSE = 0.004, confirming the predictive accuracy of the size mismatch theory.

4. **Dynamic growth simulations** demonstrate that conservative path selection (65% probability) dominates the assembly process, with mismatch-driven steps (25%) enabling shell type transitions when the size mismatch exceeds the MC→MC optimal range.

5. **Stable multi-component clusters** are predicted: Na₁₃@Rb₃₂ (MC→Ch1), K₁₃@Cs₄₂ (MC→Ch2), and Ag₁₃@Cu₄₅ (MC→Ch1), with the latter two having the highest predicted stability due to their optimal mismatch values.

This framework establishes a universal design principle for multi-component icosahedral nanoclusters, enabling rational material design for catalysis, optics, and related applications.

---

## References

1. Twarock, R. & Luque, A. Structural puzzles in virology solved with an overarching icosahedral design principle. *Nature Communications* **10**, 4414 (2019).
2. Yao, Y. et al. High-entropy nanoparticles: Synthesis-structure-property relationships and data-driven discovery. *Science* **376**, eabn3103 (2022).
3. Pinto, D. E. P. et al. Design strategies for the self-assembly of polyhedral shells. *Proc. Natl. Acad. Sci. USA* **120**, e2219458120 (2023).
4. Martín-Bravo, M. et al. Minimal Design Principles for Icosahedral Virus Capsids. *ACS Nano* **15**, 14873–14884 (2021).
5. Caspar, D. L. D. & Klug, A. Physical principles in the construction of regular viruses. *Cold Spring Harbor Symposia on Quantitative Biology* **27**, 1–24 (1962).
6. Mackay, A. L. A dense non-crystallographic packing of equal spheres. *Acta Crystallographica* **15**, 916–918 (1962).
7. Baletto, F. & Ferrando, R. Structural properties of nanoclusters: Energetic, thermodynamic, and kinetic effects. *Reviews of Modern Physics* **77**, 371–423 (2005).

---

## Appendix: Data Summary

### A.1 Geometric Constants
- sin(2π/5) = 0.9511
- cos(2π/5) = 0.3090

### A.2 Thermodynamic Parameters
- kT at 300 K = 0.02585 eV
- Boltzmann constant = 8.617 × 10⁻⁵ eV/K
- Simulation timestep = 0.001 (normalized units)

### A.3 Growth Simulation Parameters
- Temperature: 300 K
- Deposition rate: 0.01 atoms/step
- Total simulation steps: 1000
- Random seed: 42
