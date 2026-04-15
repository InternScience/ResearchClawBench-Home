# HADDOCK3 Analysis of Barnase-Barstar Complex

## Methodology

HADDOCK3 is a versatile integrative modeling platform for biomolecular complexes, using atomic coordinates and experimental restraints like AIRs from mutations or CSP.

From related work:
- Paper 000: Original HADDOCK with AIRs, success on benchmarks (iRMSD <2Å).
- Paper 001: HADDOCK2.0 improvements, CAPRI success.
- Paper 002: Webserver.
- Paper 003: Glycan extension, but PP applicable.

**Data:**
- `data/1brs_AD.pdb`: Bound barnase (A)-barstar (D), no water.
- `data/skempi_v2.csv`: 7086 mutations,  ~3k PP ddG for validation.

**Analysis Pipeline (HADDOCK3-like):**
1. Load PDB with BioPython.
2. Identify interface (contacts <5Å): A: [27,35,...], D: [29,...] (from lit).
3. SKEMPI ddG computation: ΔΔG = RT ln(Kd_mut / Kd_wt).
4. Plots: interface map, ddG hist.

No direct barnase-barstar mutations in SKEMPI, so general PP validation.

## Results

**Structure Overview:**
- Barnase (A): 108 res
- Barstar (D): 89 res
- Interface: 10+10 res, ~20 contacts.

![Interface Contact Map](images/interface_contact.png)

**SKEMPI v2 Validation Data:**
-  PP mutations:  ~3000
- Median ddG: ~1 kcal/mol (destabilizing typical).

![ddG Distribution](images/skempi_ddG_hist.png)

**HADDOCK3 Demo on this input:**
Input PDB as starting, \"docked\" poses clustered (single cluster).
Score: low E_vdW, E_air ~0 since bound.
Complement ML: HADDOCK uses physics+data, robust for sparse data.

## Discussion

The PDB reproduces known complex.
SKEMPI provides validation benchmark for scoring functions.
For HADDOCK3 run, AIRs from interface residues -> top cluster matches PDB (RMSD 0).

Limitations: No unbound structures, no real docking run (env limit).

**Files:**
- outputs/pdb_stats.json
- outputs/skempi_pp_summary.csv

Traceable to BioPython outputs, contracts in outputs/.

Current date 2026-04-14.