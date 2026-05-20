# Research Plan: Unified Diffusion Framework for Biomolecular Complex Structure Prediction

## Phase 1: Data Exploration & Parsing
- [Y] Explore workspace structure
- [ ] Parse PDB file, extract CA coordinates and residue info
- [ ] Parse SDF file, extract ligand atomic coordinates and bonds
- [ ] Analyze structural properties of FKBP12 and FK506

## Phase 2: Diffusion Framework Design
- [ ] Implement SE(3)-equivariant diffusion process
- [ ] Build forward diffusion (noising) for protein CA atoms and ligand atoms
- [ ] Build reverse diffusion (denoising) network architecture
- [ ] Implement loss functions (FAPE, distogram, etc.)

## Phase 3: Implementation & Simulation
- [ ] Implement protein feature extraction
- [ ] Implement ligand featurization
- [ ] Create diffusion sampling code
- [ ] Run forward diffusion and reverse denoising demo

## Phase 4: Analysis & Figures
- [ ] Protein structure visualization (3D scatter)
- [ ] Ligand structure visualization
- [ ] Diffusion trajectory visualization
- [ ] RMSD analysis
- [ ] Distance matrix analysis
- [ ] Method comparison figures

## Phase 5: Report Writing
- [ ] Write report/report.md with all sections
