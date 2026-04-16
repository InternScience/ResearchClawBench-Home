# Research Plan

## Task Description
Create a multi-source, hierarchical time-series dataset (HEEW) containing electricity, heat, cooling loads, PV generation, GHG emissions, and 7 weather attributes from 2014 to 2022. The data is organized hierarchically (buildings -> community -> total).
**Note:** The provided dataset `data/HEEW_Mini-Dataset` is a smaller version for 2014, covering 10 buildings (BN001-BN010), 1 community (CN01), and the total area (Total), along with weather data (`Total_weather.csv`).

## Steps
1. **Data Loading & Exploration:**
   - Load building energy data (`BN001` to `BN010`).
   - Load community and total energy data (`CN01`, `Total`).
   - Load weather data (`Total_weather.csv`).
   - Verify timestamps, data types, and missing values.

2. **Data Cleaning & Preprocessing:**
   - Handle missing values (imputation).
   - Align timestamps across all files.
   - Merge energy data with weather data.

3. **Consistency Verification (Hierarchical Aggregation):**
   - Verify if the sum of 10 buildings (`BN001` to `BN010`) equals the community (`CN01`).
   - Verify if the community (`CN01`) equals the total (`Total`), or understand the hierarchy.

4. **Analysis & Visualization:**
   - Correlation analysis between energy variables and weather attributes.
   - Time-series plots of energy consumption/generation (daily/monthly profiles).
   - Hierarchical consistency plots.

5. **Report Generation:**
   - Write methodology, results, and discussion in `report/report.md`.
   - Include generated figures in `report/images/`.

