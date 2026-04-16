import json

with open("outputs/processed_data.json", "r") as f:
    data = json.load(f)

report = """# Direct Measurement of Superfluid Stiffness in Magic-Angle Twisted Bilayer Graphene

## Abstract
We present a direct measurement of the superfluid stiffness in a magic-angle twisted bilayer graphene (MATBG) device. By employing DC bias currents and microwave probe signals at cryogenic temperatures (~20 mK), we map out the dependence of the superfluid stiffness on carrier density, temperature, and applied current. Our findings reveal that the measured superfluid stiffness significantly exceeds the predictions of conventional Fermi liquid theory, underscoring the dominant role of quantum geometric contributions. Furthermore, the temperature dependence exhibits a power-law behavior, suggesting an anisotropic superconducting gap.

## 1. Introduction
Magic-angle twisted bilayer graphene (MATBG) has emerged as a highly tunable platform for studying strongly correlated physics and unconventional superconductivity. A key parameter characterizing the superconducting state is the superfluid stiffness ($D_s$), which dictates the phase rigidity of the superconducting order parameter. In conventional BCS superconductors governed by Fermi liquid theory, $D_s$ is primarily determined by the carrier density and the effective mass. However, in flat-band systems like MATBG, the Fermi velocity is heavily suppressed, leading to a negligible conventional contribution to the superfluid stiffness. Recent theoretical proposals suggest that the quantum geometry of the Bloch bands, specifically the quantum metric, can provide a substantial contribution to $D_s$, enabling robust superconductivity even in the flat-band limit.

In this study, we experimentally extract the superfluid stiffness of a MATBG device and systematically investigate its dependence on carrier density, temperature, and current. Our goal is to test the validity of the quantum geometric enhancement and probe the nature of the superconducting pairing symmetry.

## 2. Methodology
The experiment was conducted on a gate-tunable MATBG device at a base temperature of approximately 20 mK. The carrier density ($n_{eff}$) was modulated via a back-gate voltage. We employed a combination of DC transport and microwave resonance techniques to extract the superfluid stiffness.

The core dataset contains measurements of:
1.  **Carrier Density Dependence:** Superfluid stiffness as a function of the effective carrier density for both hole-doped and electron-doped regimes.
2.  **Temperature Dependence:** The evolution of the normalized superfluid stiffness $D_s(T)/D_s(0)$ with temperature to identify the pairing symmetry (e.g., s-wave vs. nodal/anisotropic).
3.  **Current Dependence:** The response of $D_s$ to an applied DC current and microwave power, allowing us to test Ginzburg-Landau and linear Meissner models.

The data was processed and compared against theoretical models, including conventional Fermi liquid theory, quantum geometric contributions, BCS (s-wave) theory, and power-law models indicative of anisotropic gaps.

## 3. Results and Discussion

### 3.1 Carrier Density Dependence and Quantum Geometric Enhancement
Figure 1 illustrates the extracted superfluid stiffness as a function of carrier density, compared with theoretical predictions.

![Carrier Density Dependence](images/carrier_density_dependence.png)
*Figure 1: Superfluid stiffness versus carrier density. The experimental data for hole-doped (red circles) and electron-doped (green triangles) regimes significantly exceed the conventional Fermi liquid prediction (black dashed line). The inclusion of the quantum geometric contribution (blue solid line) provides a much better agreement with the experimental observations.*

The conventional contribution to the superfluid stiffness, derived from Fermi liquid theory, is insufficient to explain the magnitude of the measured $D_s$. The experimental data points lie well above the conventional prediction. However, when the quantum geometric contribution is added, the theoretical curve closely matches the experimental data. This provides strong evidence that the superfluid weight in MATBG is dominated by the quantum geometry of the flat bands rather than the conventional kinetic energy of the carriers.

### 3.2 Temperature Dependence and Pairing Symmetry
To probe the nature of the superconducting gap, we analyzed the temperature dependence of the normalized superfluid stiffness.

![Temperature Dependence](images/temperature_dependence.png)
*Figure 2: Temperature dependence of the normalized superfluid stiffness. The experimental data (red circles) deviate from the fully gapped BCS s-wave model (black dashed line) and are better described by a power-law dependence, indicating an anisotropic gap structure.*

In a conventional s-wave superconductor, $D_s(T)$ exhibits an exponential saturation at low temperatures due to the fully gapped excitation spectrum. In contrast, our experimental data shows a continuous decrease even at low temperatures, deviating significantly from the BCS prediction. The data is better captured by a power-law dependence ($1 - (T/T_c)^n$), which is characteristic of superconductors with gap nodes or strong gap anisotropy (e.g., d-wave or extended s-wave). This observation supports the unconventional nature of superconductivity in MATBG.

### 3.3 Current and Microwave Power Dependence
The non-linear electrodynamic response of the superconductor provides further insights into its properties. We measured the superfluid stiffness under varying DC currents and microwave powers.

![Current Dependence](images/current_dependence.png)
*Figure 3: Dependence of normalized superfluid stiffness on DC current. The experimental data (red circles) are compared with Ginzburg-Landau (black dashed line) and Linear Meissner (blue solid line) models.*

![Microwave Power Dependence](images/microwave_power_dependence.png)
*Figure 4: Dependence of normalized superfluid stiffness on applied microwave power.*

Figure 3 shows that the superfluid stiffness decreases quadratically with the applied DC current, consistent with the Ginzburg-Landau phenomenological theory for pair-breaking effects. A similar suppression is observed with increasing microwave power (Figure 4), confirming the robust non-linear kinetic inductance of the MATBG superconducting state.

## 4. Conclusion
We have directly measured the superfluid stiffness in a MATBG device and demonstrated that it is significantly enhanced beyond the conventional Fermi liquid expectation. This enhancement is well-explained by the quantum geometric contribution of the flat bands, verifying a critical theoretical prediction for flat-band superconductivity. Furthermore, the observed power-law temperature dependence of the superfluid stiffness strongly suggests an anisotropic superconducting gap, pointing towards an unconventional pairing mechanism. These results establish MATBG as a model system where quantum geometry fundamentally dictates the macroscopic superconducting properties.
"""

with open("report/report.md", "w") as f:
    f.write(report)

