# Research Plan: Constraining Ultralight Bosons via Black Hole Superradiance

## 1. Goal
Develop and apply a novel Bayesian statistical framework to constrain the properties of ultralight bosons (ULBs), specifically their mass ($\mu$) and self-interaction coupling strength, using posterior distributions of black hole mass ($M$) and spin ($a^*$). Produce a comprehensive `report/report.md`.

## 2. Methodology
- **Superradiance Physics**: Ultralight bosons can extract energy and angular momentum from rotating black holes if their Compton wavelength is comparable to the black hole size ($r_g \sim \hbar / (\mu c)$). This process spins down the black hole, creating a "gap" in the $M - a^*$ plane where rapidly spinning black holes should not exist if a boson of mass $\mu$ exists.
- **Bayesian Framework**: Given posterior samples of $(M, a^*)$ for a specific black hole, we want to calculate the likelihood or exclusion probability of a given boson mass $\mu$.
  - For a given $\mu$ and $(M, a^*)$, we can compute the superradiance timescale $\tau_{SR}$.
  - If $\tau_{SR}$ is much shorter than the typical age of the black hole or accretion timescale (e.g., $\tau_{age} \sim 10^7 - 10^9$ years for SMBH, $10^6$ years for stellar BH), then the observed high spin $(M, a^*)$ state is incompatible with the existence of the boson.
  - We will integrate this exclusion condition over the posterior samples of $(M, a^*)$ to get a robust exclusion probability for each $\mu$.
- **Data**:
  - M33 X-7 (Stellar-mass BH): Probes $\mu \sim 10^{-12} - 10^{-11}$ eV.
  - IRAS 09149-6206 (Supermassive BH): Probes $\mu \sim 10^{-19} - 10^{-18}$ eV.

## 3. Steps
- [ ] **Step 1: Superradiance Rate Formula**: Extract the precise formula for the superradiance rate $\Gamma_{SR}$ (or timescale $\tau_{SR} = 1/\Gamma_{SR}$) as a function of $M, a^*, \mu$ from the related work.
- [ ] **Step 2: Bayesian Exclusion Model**: Define the exclusion probability $P_{excl}(\mu)$ given the posterior samples. E.g., $P_{excl}(\mu) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\tau_{SR}(M_i, a^*_i, \mu) < \tau_{age})$.
- [ ] **Step 3: Implementation**: Write Python code to compute $P_{excl}(\mu)$ over a grid of $\mu$ values for both datasets.
- [ ] **Step 4: Visualization**: Plot the $(M, a^*)$ posterior samples against the superradiance exclusion contours. Plot the exclusion probability $P_{excl}(\mu)$ vs $\mu$.
- [ ] **Step 5: Report Writing**: Draft `report/report.md` with methodology, results, and discussion.
