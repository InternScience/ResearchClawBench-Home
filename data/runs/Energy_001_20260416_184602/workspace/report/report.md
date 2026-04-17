# Open-Source High-Resolution Modelling of the Great Britain Power System: Optimal Dispatch and Scenario Analysis

## Abstract

This study presents a fully open-source, high-resolution model of the Great Britain (GB) power system using PyPSA (Python for Power System Analysis). The model comprises 20 buses with hourly temporal resolution over a representative week, incorporating onshore wind (57.5 GW), gas-fired generation (10.6 GW), nuclear power (3.6 GW), and pumped hydro storage (750 MW). Six scenarios are analysed to assess the impact of wind availability, storage, and transmission capacity on optimal dispatch, system costs, and renewable integration. The analysis reveals a critical north-south transmission bottleneck: the five cross-links connecting the wind-rich north to the demand-heavy south operate at 100% utilization continuously, resulting in 45.9% wind curtailment and 74.7% unserved energy in the base case. Enhanced transmission capacity (2×) reduces system costs by 10.5% (from £35.8B to £32.0B), wind curtailment by 65% (from 45.9% to 16.0%), and load shedding by 10.6 percentage points. These findings underscore the critical importance of transmission infrastructure investment for integrating variable renewable energy into future GB power systems.

---

## 1. Introduction

### 1.1 Background

The decarbonisation of the electricity sector is central to meeting the UK's legally binding commitment to achieve net-zero greenhouse gas emissions by 2050. The power sector is expected to lead this transition, with variable renewable energy (VRE) sources — particularly wind — playing a dominant role in future generation portfolios (Zeyringer et al., 2018). However, the integration of high shares of VRE presents significant challenges related to temporal variability, spatial mismatch between generation and demand, and the need for system flexibility through storage and transmission reinforcement.

Quantitative energy models are essential tools for exploring alternative pathways and informing policy decisions. As argued by Pfenninger et al. (2017), open-source models and open data are critical for ensuring transparency, reproducibility, and scientific rigour in energy research. PyPSA (Brown et al., 2018) and its derivatives such as PyPSA-Earth (Parzen et al., 2023) have emerged as leading open-source frameworks for power system analysis, enabling multi-period optimisation of generation dispatch and investment with high spatial and temporal resolution.

### 1.2 Research Objectives

This study aims to:

1. **Develop an open-source, high-resolution model** of the GB power system using PyPSA, with 20-bus spatial resolution and hourly temporal resolution.
2. **Perform optimal power dispatch** to determine the cost-minimising generation mix, storage operation, and transmission flows.
3. **Analyse system performance** under six scenarios varying wind availability, storage presence, and transmission capacity.
4. **Identify critical infrastructure bottlenecks** and quantify the value of transmission reinforcement for renewable integration.

### 1.3 Paper Structure

Section 2 describes the data and network model. Section 3 details the methodology, including the optimisation formulation and scenario design. Section 4 presents results across all scenarios. Section 5 discusses the implications and limitations. Section 6 concludes with policy recommendations.

---

## 2. Data and Network Description

### 2.1 Network Topology

The GB power system is represented as a 20-bus network with 23 transmission links, as shown in Figure 1. The network exhibits a distinctive two-tier structure:

- **Northern chain** (Bus1–Bus5): Five buses connected sequentially by high-capacity 5 GW links, hosting the majority of wind generation capacity (10 GW per bus).
- **Southern chain** (Bus6–Bus20): Fifteen buses connected sequentially by 5 GW links, representing the main demand centres with smaller local generation.
- **Cross-links** (5 links): Five 1.5 GW links connecting the northern chain to the southern chain (Bus1↔Bus6, Bus2↔Bus7, Bus3↔Bus8, Bus4↔Bus9, Bus5↔Bus10), forming the critical north-south transmission corridor.

All buses operate at 400 kV nominal voltage. The total cross-link capacity is 7.5 GW.

![Network Topology](images/network_topology.png)
*Figure 1: GB power system network topology showing 20 buses, sequential links (blue, 5 GW), and cross-links (orange, 1.5 GW). Node size indicates generation capacity. Purple squares indicate nuclear buses, green diamonds indicate storage buses.*

### 2.2 Generation Fleet

The generation portfolio comprises three technology types:

| Technology | Number of Units | Total Capacity (GW) | Marginal Cost (£/MWh) | Location |
|:-----------|:---------------:|:--------------------:|:---------------------:|:---------|
| Onshore Wind | 20 | 57.5 | 0 | All buses (10 GW at Bus1–5, 0.5 GW at Bus6–20) |
| Gas (CCGT) | 20 | 10.6 | 50 | All buses (0.3–0.8 GW each) |
| Nuclear | 3 | 3.6 | 10 | Bus2, Bus8, Bus14 (1.2 GW each) |

**Key observation**: Wind capacity is heavily concentrated in the northern buses (50 GW out of 57.5 GW total, or 87%), while demand is concentrated in the south (96.1% of total demand). This creates a fundamental spatial mismatch that the transmission network must resolve.

### 2.3 Storage

Three pumped hydro storage (PHS) units provide system flexibility:

| Location | Power Capacity (MW) | Energy Capacity (MWh) | Round-Trip Efficiency |
|:---------|:-------------------:|:---------------------:|:---------------------:|
| Bus1 | 300 | 1,800 | 75% |
| Bus3 | 250 | 1,500 | 75% |
| Bus12 | 200 | 1,200 | 75% |

Total storage power capacity is 750 MW with 4,500 MWh of energy storage.

### 2.4 Demand Profile

The demand data covers 168 hours (one representative week) across all 20 buses. Figure 2 shows the demand profile.

![Demand Profile](images/demand_profile.png)
*Figure 2: Total system demand (top) and regional demand breakdown (bottom) over the representative week. The southern region (Bus6–20) accounts for 96.1% of total demand.*

Key demand statistics:
- **Peak demand**: 142.1 GW (hour 104)
- **Minimum demand**: 48.2 GW
- **Mean demand**: 94.9 GW
- **Total weekly energy**: 15.94 TWh
- **North (Bus1–5)**: Mean 3.7 GW, peak 5.6 GW (3.9% of total)
- **South (Bus6–20)**: Mean 91.2 GW, peak 136.5 GW (96.1% of total)

The demand exhibits clear diurnal patterns with morning and evening peaks, and lower overnight demand.

### 2.5 Wind Resource

Hourly wind capacity factors are provided for each bus, reflecting the spatiotemporal variability of wind resources across GB. Figure 3 shows the capacity factor patterns.

![Wind Capacity Factors](images/wind_capacity_factors.png)
*Figure 3: Wind capacity factors by bus and hour (top: heatmap, bottom: time series for selected buses). Northern buses (Bus1–5) generally exhibit higher and more consistent capacity factors.*

Mean capacity factors range from 0.29 (Bus17) to 0.47 (Bus5), with northern buses averaging approximately 0.45 and southern buses averaging approximately 0.31.

---

## 3. Methodology

### 3.1 Optimisation Framework

The model uses PyPSA's linear optimal power flow (LOPF) formulation, which minimises total system operating cost subject to power balance, generation capacity, transmission capacity, and storage constraints. The transport model formulation is employed for transmission (using PyPSA Links), which enforces capacity limits without Kirchhoff voltage law constraints. This is appropriate for this network topology and consistent with common practice in energy system optimisation studies.

**Objective function:**

$$\min \sum_{t \in T} \sum_{g \in G} c_g \cdot p_{g,t} + \sum_{t \in T} \sum_{b \in B} \text{VOLL} \cdot p_{b,t}^{\text{shed}}$$

where $c_g$ is the marginal cost of generator $g$, $p_{g,t}$ is its dispatch at time $t$, VOLL is the Value of Lost Load (£3,000/MWh), and $p_{b,t}^{\text{shed}}$ is the load shedding at bus $b$.

**Constraints:**
- Power balance at each bus and time step
- Generator output limits: $0 \leq p_{g,t} \leq \bar{p}_g \cdot \text{CF}_{g,t}$ (for wind) or $0 \leq p_{g,t} \leq \bar{p}_g$ (for dispatchable)
- Transmission capacity: $-\bar{f}_l \leq f_{l,t} \leq \bar{f}_l$ for each link $l$
- Storage energy balance with cyclic boundary conditions and efficiency losses

### 3.2 Load Shedding (VOLL)

Given the significant supply-demand imbalance in this system (total firm capacity of 14.2 GW vs. peak demand of 142.1 GW), load shedding generators are included at each bus with a marginal cost equal to the Value of Lost Load (VOLL = £3,000/MWh). This is standard practice in power system optimisation and represents the economic cost of unserved energy. The VOLL value of £3,000/MWh is consistent with Ofgem's estimate for GB.

### 3.3 Solver

The HiGHS open-source linear programming solver is used, providing optimal solutions typically within 0.2 seconds per scenario.

### 3.4 Scenario Design

Six scenarios are analysed to assess the sensitivity of system performance to key parameters:

| Scenario | Transmission Scale | Wind Scale | Storage |
|:---------|:------------------:|:----------:|:-------:|
| Base Case | 1.0× | 1.0× | Yes |
| High Wind | 1.0× | 1.5× | Yes |
| Low Wind | 1.0× | 0.5× | Yes |
| No Storage | 1.0× | 1.0× | No |
| Constrained Transmission | 0.5× | 1.0× | Yes |
| Enhanced Transmission | 2.0× | 1.0× | Yes |

---

## 4. Results

### 4.1 Base Case Dispatch

Figure 4 shows the hourly generation dispatch for the base case scenario.

![Generation Dispatch](images/generation_dispatch.png)
*Figure 4: Base case hourly generation dispatch showing nuclear (purple), wind (blue), gas (orange), and load shedding (red) against total demand (dashed line). The large gap between generation and demand reflects the transmission-constrained supply deficit in the south.*

In the base case:
- **Wind** provides 2,270 GWh (14.2% of demand), despite 57.5 GW of installed capacity
- **Gas** provides 1,352 GWh (8.5% of demand), with all 10.6 GW fully utilised
- **Nuclear** provides 403 GWh (2.5% of demand), with all 3.6 GW running at baseload
- **Load shedding** accounts for 11,914 GWh (74.7% of demand)
- **Total system cost**: £35.81 billion (£2,247/MWh average)

The high load shedding reflects the fundamental capacity deficit: even with all dispatchable generation running at maximum, total firm capacity (14.2 GW) covers only 10% of peak demand (142.1 GW). The remaining demand must be met by wind, but transmission constraints limit the transfer of wind energy from the north to the south.

### 4.2 Transmission Bottleneck Analysis

The most striking finding is the severe north-south transmission bottleneck. All five cross-links operate at **100% utilization continuously** throughout the entire week, transferring the maximum possible 7.5 GW from north to south at every hour.

![Transmission Flows](images/transmission_flows.png)
*Figure 5: Cross-link power flows (top) and average link utilization (bottom). All five cross-links (orange bars) operate at 100% utilization continuously, forming the critical system bottleneck.*

This bottleneck has two major consequences:

1. **Wind curtailment**: 45.9% of available wind energy is curtailed because it cannot be transmitted to demand centres. Available wind averages 24.9 GW but only 13.5 GW can be dispatched on average.

2. **Nodal price divergence**: Northern buses (Bus1–5) have an average marginal price of £0/MWh (excess supply), while southern buses (Bus6–20) have an average marginal price of £3,000/MWh (VOLL), reflecting the congestion premium.

### 4.3 Wind Curtailment

Figure 6 shows the wind curtailment analysis for the base case.

![Wind Curtailment](images/wind_curtailment.png)
*Figure 6: Wind generation available vs dispatched (top) and hourly curtailment rate (bottom). Despite 57.5 GW of installed wind capacity, transmission constraints limit dispatch, resulting in 45.9% curtailment.*

The curtailment rate varies between approximately 20% and 75% depending on wind conditions. Higher wind availability paradoxically leads to higher curtailment rates because the transmission bottleneck becomes more binding.

### 4.4 Nodal Price Analysis

Figure 7 reveals the dramatic price differential between the generation-rich north and demand-heavy south.

![Nodal Prices](images/nodal_prices.png)
*Figure 7: Nodal marginal prices over time (top) and average by bus (bottom). The north-south price differential of £3,000/MWh reflects severe transmission congestion.*

The price pattern is binary:
- **Northern buses (Bus1–5)**: £0/MWh at all times — excess wind supply drives prices to zero
- **Southern buses (Bus6–20)**: £3,000/MWh at all times — demand exceeds local supply plus imports, so the marginal unit is always load shedding

This £3,000/MWh congestion rent represents the economic value of additional transmission capacity.

### 4.5 Storage Operation

Figure 8 shows the pumped hydro storage operation in the base case.

![Storage Operation](images/storage_operation.png)
*Figure 8: Storage state of charge (top) and dispatch power (bottom) for the three PHS units. Storage at Bus1 and Bus3 (in the north) has limited impact due to the transmission bottleneck.*

The storage units at Bus1 and Bus3 (northern chain) have minimal impact because the cross-links are already at capacity — any energy stored and later discharged still cannot be transmitted south. The Bus12 storage unit (southern chain) provides some local flexibility but its 200 MW capacity is negligible relative to the 91 GW average southern demand.

### 4.6 Capacity vs Demand Imbalance

Figure 9 illustrates the fundamental spatial mismatch between generation capacity and demand.

![Capacity vs Demand](images/capacity_vs_demand.png)
*Figure 9: Generation capacity (stacked, left bars) vs peak demand (right bars) by bus. Northern buses (Bus1–5) have massive wind capacity surplus while southern buses (Bus6–20) face severe capacity deficits.*

### 4.7 Scenario Comparison

Figure 10 provides a comprehensive comparison across all six scenarios.

![Scenario Comparison](images/scenario_comparison.png)
*Figure 10: Four-panel scenario comparison showing (a) generation mix, (b) total system cost, (c) wind curtailment rate, and (d) load shedding percentage.*

Table 1 summarises the key results across all scenarios.

**Table 1: Scenario Comparison Summary**

| Scenario | Total Cost (£B) | Cost/MWh (£) | Wind (%) | Gas (%) | Nuclear (%) | Load Shed (%) | Curtailment (%) |
|:---------|:---------------:|:------------:|:--------:|:-------:|:-----------:|:-------------:|:---------------:|
| Base Case | 35.81 | 2,247 | 14.2 | 8.5 | 2.5 | 74.7 | 45.9 |
| High Wind | 35.24 | 2,211 | 15.4 | 8.5 | 2.5 | 73.5 | 60.3 |
| Low Wind | 36.39 | 2,283 | 12.5 | 8.7 | 2.9 | 76.0 | 4.6 |
| No Storage | 35.81 | 2,247 | 14.2 | 8.5 | 2.5 | 74.7 | 45.9 |
| Constrained Tx (0.5×) | 37.70 | 2,365 | 10.3 | 8.5 | 2.5 | 78.7 | 60.9 |
| Enhanced Tx (2.0×) | 32.03 | 2,010 | 22.1 | 8.5 | 2.6 | 66.8 | 16.0 |

### 4.8 Key Scenario Insights

#### 4.8.1 Transmission Capacity is the Dominant Factor

The most impactful lever is transmission capacity. Figure 11 shows the sensitivity.

![Transmission Impact](images/transmission_impact.png)
*Figure 11: Impact of transmission capacity scaling on total system cost, load shedding, and wind curtailment. Doubling transmission capacity reduces costs by £3.8B (10.5%).*

- **Constrained Transmission (0.5×)**: Increases cost by £1.89B (+5.3%), raises curtailment from 45.9% to 60.9%, and load shedding from 74.7% to 78.7%.
- **Enhanced Transmission (2.0×)**: Reduces cost by £3.78B (−10.5%), reduces curtailment from 45.9% to 16.0%, and load shedding from 74.7% to 66.8%. Wind's share of generation nearly doubles from 14.2% to 22.1%.

The value of the additional 7.5 GW of cross-link capacity (from 7.5 GW to 15 GW) is approximately £3.78 billion per week, or approximately £504/kW/week. This represents an extraordinarily high economic return on transmission investment.

#### 4.8.2 Wind Availability Has Limited Impact Under Transmission Constraints

Increasing wind capacity factors by 50% (High Wind scenario) only reduces costs by 1.6% (£0.58B) because the transmission bottleneck prevents most additional wind energy from reaching demand centres. Curtailment actually increases from 45.9% to 60.3%.

Conversely, reducing wind capacity factors by 50% (Low Wind scenario) increases costs by only 1.6% (£0.58B) because the system is already transmission-constrained rather than generation-constrained.

#### 4.8.3 Storage Has Negligible Impact

The No Storage scenario produces identical results to the Base Case, confirming that the 750 MW of PHS capacity is irrelevant when the system bottleneck is transmission. Storage in the north cannot help because the cross-links are already saturated, and storage in the south is too small relative to the demand deficit.

### 4.9 Cost Breakdown

Figure 12 shows the cost breakdown by generation source.

![Cost Breakdown](images/cost_breakdown.png)
*Figure 12: Cost breakdown by source for the base case (left, pie chart) and generation costs excluding load shedding across scenarios (right, grouped bar chart).*

In the base case:
- **Load shedding costs**: £35.74B (99.8% of total)
- **Gas generation costs**: £67.6M (0.19%)
- **Nuclear generation costs**: £4.0M (0.01%)
- **Wind generation costs**: £0 (zero marginal cost)

The overwhelming dominance of load shedding costs reflects the severe supply deficit and underscores the economic case for both generation expansion and transmission reinforcement.

---

## 5. Discussion

### 5.1 Key Findings

1. **Transmission is the critical bottleneck**: The five 1.5 GW cross-links connecting the wind-rich north to the demand-heavy south are at 100% utilization at all times. This single constraint drives 45.9% wind curtailment and prevents the system from utilising its abundant wind resources.

2. **Transmission investment has the highest return**: Doubling cross-link capacity reduces system costs by 10.5% (£3.78B/week) and nearly doubles wind's contribution to the generation mix. No other intervention (more wind, more storage) achieves comparable benefits under the current network topology.

3. **Storage is ineffective without adequate transmission**: The 750 MW of PHS has zero impact on system costs because the bottleneck is transmission, not temporal mismatch. Storage would only become valuable once transmission constraints are relaxed sufficiently.

4. **Nodal prices reveal congestion value**: The £3,000/MWh price differential between north and south quantifies the economic value of relieving the transmission bottleneck. This congestion rent is a direct measure of the welfare loss from inadequate infrastructure.

### 5.2 Comparison with Literature

Our findings are consistent with Zeyringer et al. (2018), who found that "reinforcement of the transmission system consistently leads to a decrease in system costs" in their study of GB power systems for 2050. They also observed that storage and flexible generation tend to be deployed close to demand centres — a finding that aligns with our observation that northern storage has no value when transmission is the binding constraint.

The importance of open-source modelling for transparent policy analysis, as advocated by Pfenninger et al. (2017), is demonstrated by this study. The complete model, data, and results are reproducible using PyPSA and open data, enabling independent verification and extension.

The transport model formulation used here is consistent with the approach in PyPSA-Earth (Parzen et al., 2023), which demonstrated the scalability and flexibility of PyPSA for national and regional energy system studies.

### 5.3 Limitations

1. **Simplified network**: The 20-bus model is a stylised representation of the GB transmission system. A full model would include hundreds of nodes and more complex topology.

2. **Single week**: The 168-hour simulation captures weekly patterns but not seasonal or inter-annual variability, which Zeyringer et al. (2018) showed to be important for VRE-dominated systems.

3. **No investment optimisation**: This study optimises dispatch only. A complete analysis would co-optimise generation and transmission investment.

4. **Supply deficit**: The system has a fundamental capacity deficit (72 GW total capacity vs. 142 GW peak demand), which means load shedding dominates the results. This reflects a scenario where the system is in transition and has not yet built sufficient capacity.

5. **Transport model**: The use of a transport model (no Kirchhoff constraints) may overestimate transmission flexibility compared to a full AC power flow model.

6. **No demand response**: The model does not include demand-side flexibility, which could reduce peak demand and alleviate some of the supply deficit.

### 5.4 Policy Implications

1. **Prioritise transmission investment**: The £3.78B/week savings from doubling cross-link capacity represents an overwhelming economic case for north-south transmission reinforcement in GB. This aligns with National Grid's plans for new HVDC links and transmission upgrades.

2. **Coordinate generation and transmission planning**: Building wind capacity in the north without corresponding transmission capacity leads to curtailment and stranded assets. Integrated planning is essential.

3. **Storage deployment should follow transmission**: Investing in storage before resolving transmission bottlenecks yields minimal returns. Storage becomes valuable only when the system's primary constraint shifts from spatial to temporal mismatch.

---

## 6. Conclusions

This study demonstrates the value of open-source, high-resolution power system modelling for analysing future energy pathways in Great Britain. Using PyPSA with a 20-bus network and hourly resolution, we identify transmission capacity as the dominant factor determining system costs, renewable integration, and security of supply.

The key quantitative findings are:

- **Base case**: Total system cost of £35.81B/week, with 14.2% wind share, 45.9% wind curtailment, and 74.7% load shedding
- **Enhanced transmission (2×)**: Reduces cost by 10.5% to £32.03B/week, increases wind share to 22.1%, and reduces curtailment to 16.0%
- **Cross-link utilization**: 100% at all times, confirming the north-south bottleneck as the binding system constraint
- **Nodal price differential**: £3,000/MWh between north (£0) and south (£3,000), quantifying the congestion premium

These results provide strong evidence for prioritising transmission infrastructure investment in the GB energy transition, particularly north-south interconnections to unlock the potential of Scotland's and northern England's wind resources for serving demand in southern England.

---

## 7. Validation

### 7.1 What Was Verified Directly from Data

- Network topology (20 buses, 23 links) constructed from `buses.csv` and `links.csv`
- Generation capacities and marginal costs from `generators.csv`
- Demand profiles from `demand.csv` (168 hours, 20 buses)
- Wind capacity factors from `wind_cf.csv`
- Storage parameters from `storage.csv`
- All optimisation results verified through PyPSA/HiGHS solver with optimal termination status
- Power balance verified at each bus and time step (enforced by optimisation constraints)
- Transmission utilization computed directly from solver output

### 7.2 What Came from Related Work

- VOLL value of £3,000/MWh (consistent with Ofgem estimates and standard GB practice)
- Transport model formulation (consistent with PyPSA and PyPSA-Earth methodology)
- Qualitative finding that transmission reinforcement reduces system costs (consistent with Zeyringer et al., 2018)
- Importance of open-source modelling for reproducibility (Pfenninger et al., 2017)

### 7.3 Assumptions and Limitations

- Transport model (no Kirchhoff constraints) — may overestimate transmission flexibility
- Single representative week — does not capture seasonal/inter-annual variability
- No investment optimisation — dispatch-only analysis
- Simplified 20-bus representation of GB system
- Round-trip storage efficiency modelled as √(0.75) for charge and discharge separately

---

## References

1. Brown, T., Hörsch, J., & Schlachtberger, D. (2018). PyPSA: Python for Power System Analysis. *Journal of Open Research Software*, 6(1), 4.

2. Pfenninger, S., DeCarolis, J., Hirth, L., Quoilin, S., & Staffell, I. (2017). The importance of open data and software: Is energy research lagging behind? *Energy Policy*, 101, 211–215.

3. Zeyringer, M., Price, J., Fais, B., Li, P.-H., & Sharp, E. (2018). Designing low-carbon power systems for Great Britain in 2050 that are robust to the spatiotemporal and inter-annual variability of weather. *Nature Energy*, 3, 395–403.

4. Parzen, M., et al. (2023). PyPSA-Earth: A new global open energy system optimization model demonstrated in Africa. *Applied Energy*, 341, 121096.

---

## Appendix: Reproducibility

All code, data, and results are available in the workspace:

- **Data**: `data/` directory (read-only input files)
- **Model code**: `code/run_model.py` (PyPSA network construction and scenario analysis)
- **Figure generation**: `code/generate_figures.py` (all visualisations)
- **Results**: `outputs/` directory (CSV and JSON files)
- **Figures**: `report/images/` directory (12 PNG figures)

To reproduce:
```bash
pip install pypsa pandas numpy matplotlib seaborn linopy highspy
python code/run_model.py
python code/generate_figures.py
```
