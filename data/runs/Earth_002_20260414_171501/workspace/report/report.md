# Composite Risk Index for Global Mangroves Under Sea-Level Rise and Tropical Cyclone Regime Pressure

## Abstract
Mangroves provide globally important coastal protection, carbon storage, and biodiversity benefits, but they are exposed to interacting climate hazards. Using only the datasets provided in the workspace, I built a transparent composite risk index that combines end-century relative sea-level-rise (RSLR) rates with a tropical-cyclone (TC) regime-pressure proxy and applied it to a 100,000-point global mangrove sample. RSLR was extracted from IPCC AR6 regional projections for SSP2-4.5, SSP3-7.0, and SSP5-8.5 using the median 2100 rate. TC pressure was represented from historical downscaled TC tracks by emphasizing major and intense storm track density in 1° cells. The resulting composite index identifies concentrated hotspots rather than uniform global risk. Estimated high-or-very-high risk mangrove area was 36.8% under SSP2-4.5, 35.9% under SSP3-7.0, and 36.4% under SSP5-8.5 in the calibrated sample-based global area reconstruction. Under SSP5-8.5, leading hotspot countries by high-or-very-high risk area included Indonesia, Mexico, Australia, Bangladesh, and the Philippines. Scenario differences were driven primarily by the SLR term, while the TC term strongly structured the upper tail of risk in cyclone-exposed regions. Because no future TC projection field was provided locally, the TC component should be interpreted as historical regime pressure rather than a full dynamical future regime-shift forecast.

## 1. Introduction
Mangroves are among the most valuable coastal ecosystems, supporting shoreline protection, carbon storage, fisheries, and biodiversity. Related work in the local workspace emphasizes two hazards that are especially relevant to their long-term persistence. First, recent synthesis work indicates that mangrove persistence becomes increasingly threatened when RSLR rates exceed approximately 4 mm/yr and highly threatened around 7 mm/yr. Second, TC literature identifies cyclones as a dominant natural disturbance for mangroves, with damage concentrated in major and especially intense storms.

The present task asks for a composite risk index that combines TC regime shifts and sea-level rise and applies it globally to assess where mangroves and their ecosystem services may be at risk by end century. The workspace contains global mangrove sample points, three IPCC AR6 RSLR scenarios, historical downscaled TC tracks, and a country boundary layer with mangrove area totals. These inputs are sufficient for a transparent, reproducible first-order global assessment, but not for a full dynamical projection of future TC regime shifts. I therefore implemented a method that preserves the requested two-hazard structure while making the deviation explicit: sea-level rise is scenario-specific and late-century, whereas the TC component is a historical regime-pressure proxy derived from major/intense storm track density.

## 2. Data overview
### 2.1 Workspace datasets used
- `data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg`: 100,000 mangrove sample points in EPSG:4326.
- `data/slr/total_ssp245_medium_confidence_rates.nc`
- `data/slr/total_ssp370_medium_confidence_rates.nc`
- `data/slr/total_ssp585_medium_confidence_rates.nc`
  - Each contains AR6 regional RSLR rates for 66,190 locations, 14 years, and 107 quantiles. I used the median quantile (0.5) and year 2100.
- `data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc`: 200,000 historical TC track records with latitude, longitude, and maximum sustained wind.
- `data/ecosystem/UCSC_CWON_countrybounds.gpkg`: country polygons with 2020 mangrove area totals used for point-to-area calibration.

### 2.2 Related work used from local PDFs
The local papers informed interpretation and method design:
- `paper_000.pdf`: coastal habitats become increasingly exposed above 4 mm/yr RSLR and highly exposed around 7 mm/yr.
- `paper_002.pdf`: TCs are a leading natural disturbance in mangroves and recurrence/intensity matter.
- `paper_003.pdf`: major cyclones dominate mangrove damage risk globally; intense cyclones are especially important.
- `paper_001.pdf`: ecosystem services should be considered in mangrove conservation prioritization.

## 3. Methods
### 3.1 Spatial unit and area calibration
The mangrove layer in the workspace is a 100,000-point sample rather than full polygons. I spatially joined points to country polygons and used the country-layer field `Mang_Ha_2020` to estimate hectares represented by each sampled point within each country:

\[
\text{ha per point}_c = \frac{\text{Mangrove area in country } c}{\text{number of sample points in country } c}
\]

Each point inherited this country-specific area weight. This produced a calibrated global mangrove area estimate of 15,394,406 ha across the sample-based reconstruction.

### 3.2 Sea-level-rise component
For each SSP scenario, I extracted the median 2100 RSLR rate from the nearest AR6 projection point to each mangrove sample point. The raw SLR variable is therefore a point-level rate in mm/yr.

To build the composite index, the scenario-specific SLR rates were normalized to 0-1 using robust min-max scaling based on the 5th and 95th percentiles of mangrove-point values.

### 3.3 Tropical-cyclone regime-pressure component
The workspace contains historical TC tracks but no explicit future regional TC change multipliers or future event catalogs. To preserve the named TC hazard in the task, I estimated a historical TC regime-pressure proxy as follows:
1. Convert TC longitudes to the [-180, 180) convention.
2. Bin records to 1° × 1° cells.
3. Count Category 3+ records using a wind threshold of 50 m/s.
4. Count Category 4+ records using a wind threshold of 58 m/s.
5. Define raw regime pressure for each mangrove point’s grid cell as:

\[
TC_{raw} = \log(1 + N_{Cat3+}) + 1.5\log(1 + N_{Cat4+})
\]

This gives additional influence to intense storms, consistent with the related-work emphasis that stronger cyclones dominate damage risk.

The raw TC measure was then normalized to 0-1 using the same robust scaling approach.

### 3.4 Composite risk index
For each scenario, the final composite risk index at each mangrove sample point was:

\[
Risk_s = 0.5 \times TC_{norm} + 0.5 \times SLR_{norm,s}
\]

where `s` is SSP245, SSP370, or SSP585.

Risk classes were assigned by global quintiles of the scenario-specific composite score:
- Very low
- Low
- Moderate
- High
- Very high

### 3.5 Validation and comparison strategy
Because no ground-truth end-century impact labels are available in the workspace, validation here is internal and literature-informed rather than predictive. I therefore used:
- scenario comparison of risk distributions,
- component relationship plots,
- direct reporting of area above 4 mm/yr and 7 mm/yr RSLR reference thresholds,
- hotspot summaries by country.

## 4. Results
### 4.1 Scenario-level direct answers
The direct scenario summary is saved in `outputs/direct_constraint_results.csv`.

Key results:
- **SSP245**: 36.85% of estimated mangrove area falls in high or very high composite risk; 17.92% is in very high risk.
- **SSP370**: 35.87% of estimated mangrove area falls in high or very high composite risk; 17.48% is in very high risk.
- **SSP585**: 36.40% of estimated mangrove area falls in high or very high composite risk; 17.48% is in very high risk.

The late-century SLR hazard is widespread in this dataset. Area above the literature-informed thresholds was:
- **Above 4 mm/yr**: 99.81% (SSP245), 100.00% (SSP370), 100.00% (SSP585)
- **Above 7 mm/yr**: 94.43% (SSP245), 99.81% (SSP370), 99.96% (SSP585)

These threshold exceedances indicate that, in the AR6 rate product provided here, most mangrove sample points lie in regions with late-century RSLR rates at or above values flagged by the related literature as ecologically consequential.

### 4.2 Geographic hotspot pattern
The global risk map (`images/global_composite_risk_map_ssp585.png`) shows that composite risk is geographically concentrated, especially in cyclone-active coasts and regions with elevated late-century SLR rates.

Under SSP5-8.5, the largest country-level high-or-very-high risk areas were estimated in:
1. Indonesia
2. Mexico
3. Australia
4. Bangladesh
5. Philippines
6. Cuba
7. Papua New Guinea
8. Mozambique
9. United States
10. India

The top-hotspot chart is shown in `images/top_country_hotspots_ssp585.png`, and the underlying table is in `outputs/country_hotspot_top20_by_scenario.csv`.

Two different hotspot modes are visible:
- **Large-area mode**: countries such as Indonesia rank highly because they have very large mangrove extent, even when mean risk is moderate rather than extreme.
- **High-intensity mode**: countries such as Mexico and the Philippines combine elevated TC pressure and high SLR, giving a high share of national mangrove area in the upper risk classes.

### 4.3 Scenario comparison
The scenario comparison figure (`images/scenario_risk_distribution.png`) shows that the global distribution of the composite index shifts only modestly across SSPs in this implementation. This is expected because:
- the TC component is held fixed across scenarios due to data limitations, and
- RSLR is already high across much of the mangrove domain even in SSP245 by 2100 in the provided AR6 rate fields.

The distribution of area by risk class was:

**SSP245**
- Very low: 24.29%
- Low: 27.89%
- Moderate: 10.97%
- High: 18.92%
- Very high: 17.92%

**SSP370**
- Very low: 20.97%
- Low: 22.24%
- Moderate: 20.91%
- High: 18.40%
- Very high: 17.48%

**SSP585**
- Very low: 22.54%
- Low: 28.66%
- Moderate: 12.40%
- High: 18.92%
- Very high: 17.48%

Because risk classes are based on quintiles of the point distribution but summarized with area weights, the shares are not exactly 20% each.

### 4.4 Relationship between components
The component comparison figure (`images/component_relationships.png`) shows that upper-tail composite risk is most common where high SLR and strong TC regime pressure coincide. In the point-level sample, the composite index correlates strongly with both components, but more strongly with the TC term in this implementation:

- SSP245: corr(composite, TC) = 0.862; corr(composite, SLR) = 0.621
- SSP370: corr(composite, TC) = 0.828; corr(composite, SLR) = 0.618
- SSP585: corr(composite, TC) = 0.857; corr(composite, SLR) = 0.656

This implies that although SLR is nearly ubiquitous at high levels by 2100, TC pressure remains important for discriminating which already-SLR-exposed mangrove regions sit in the highest composite-risk tail.

## 5. Validation, evidence, and limitations
### 5.1 Verified directly from workspace data
The following were verified directly from local artifacts:
- mangrove dataset size and geometry type,
- AR6 netCDF dimensions and presence of median quantile and year 2100,
- TC track variables and wind distribution,
- area-calibrated scenario results,
- hotspot rankings,
- generated figures and exported output tables.

### 5.2 Informed by related work
The following interpretive elements came from local related-work PDFs:
- use of 4 mm/yr and 7 mm/yr as meaningful RSLR reference levels,
- emphasis on major/intense tropical cyclones as dominant damage drivers,
- framing around ecosystem-service relevance for conservation prioritization.

### 5.3 Key limitations
1. **TC regime shifts are approximated, not dynamically projected.** The task asked for cyclone regime shifts, but the workspace does not contain future TC projection fields, future event catalogs, or region-specific scaling coefficients. I therefore used a historical regime-pressure proxy based on major/intense storm density.
2. **The mangrove layer is a sampled point dataset.** Area totals were reconstructed by country-based calibration rather than direct polygon-area calculation.
3. **Nearest-neighbor extraction introduces smoothing.** SLR rates were assigned from nearest AR6 points rather than full interpolation.
4. **No direct ecosystem-service loss model was available.** The report interprets risk to ecosystem services through likely exposure of mangrove area, not through separate service-specific response functions.
5. **Scenario contrasts are conservative in structure.** Because the TC term is held constant across SSPs, between-scenario differences arise only through the SLR component.

## 6. Implications for climate-adaptive mangrove conservation
Despite the limitations, several actionable patterns emerge.

First, the joint-hazard perspective matters. SLR alone indicates widespread late-century stress, but TC pressure distinguishes where this background stress is likely to be compounded by recurrent extreme disturbance. Second, countries with very large mangrove stocks such as Indonesia and Australia deserve attention because moderate average risk can still translate into large at-risk area. Third, regions such as Mexico and the Philippines stand out because both hazard components are elevated, implying greater urgency for adaptation strategies that strengthen resilience, reduce non-climatic stressors, and prioritize post-disturbance recovery capacity.

A practical policy interpretation is therefore to separate conservation planning into at least two tiers:
- **global stock protection** for countries with the largest absolute high-risk mangrove area, and
- **high-intensity resilience planning** for countries with the highest proportional exposure to the upper risk classes.

## 7. Reproducibility and output inventory
### Code
- `code/analyze.py`

### Main outputs
- `outputs/direct_constraint_results.csv`
- `outputs/risk_class_area_summary.csv`
- `outputs/country_hotspot_top20_by_scenario.csv`
- `outputs/mangrove_point_risk_sample.csv`
- `outputs/slr_scenario_summary.csv`
- `outputs/claim_recovery_table.csv`

### Figures
- `images/global_composite_risk_map_ssp585.png`
- `images/scenario_risk_distribution.png`
- `images/component_relationships.png`
- `images/top_country_hotspots_ssp585.png`

## 8. Conclusion
Using the local mangrove, AR6 sea-level-rise, and historical tropical cyclone datasets, I developed a reproducible composite mangrove risk index that integrates late-century RSLR with historical TC regime pressure. The analysis indicates that end-century mangrove risk is globally widespread but spatially concentrated into identifiable hotspots. In absolute area terms, Indonesia, Mexico, and Australia dominate the SSP5-8.5 hotspot ranking, while cyclone-exposed countries such as Mexico and the Philippines show especially high proportional exposure. The strongest conclusion supported by the workspace evidence is that late-century mangrove risk should be assessed as a compound hazard problem: SLR creates widespread baseline stress, and TC regime pressure helps determine where the highest-risk tail emerges.
