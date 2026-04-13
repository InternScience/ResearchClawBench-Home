# Analysis Summary

## Task-Level Scores

- Construct Kinetic Hamiltonian (continuum version, single-particle): average rubric score 1.33
- Define each term in Kinetic Hamiltonian (continuum version): average rubric score 1.67
- Construct Potential Hamiltonian (continuum version): average rubric score 1.83
- Define each term in Potential Hamiltonian (continuum version): average rubric score 2.00
- Convert from single-particle to second-quantized form, return in matrix: average rubric score 2.00
- Convert from single-particle to second-quantized form, return in summation (expand the matrix): average rubric score 1.67
- Convert noninteracting Hamiltonian in real space to momentum space (continuum version): average rubric score 1.83
- Particle-hole transformation: average rubric score 1.67
- Simplify the Hamiltonian in the particle-hole basis: average rubric score 1.83
- Construct interaction Hamiltonian (momentum space): average rubric score 2.00
- Wick's theorem: average rubric score 1.67
- Extract quadratic term: average rubric score 1.67
- Swap the index to combine Hartree and Fock terms: average rubric score 1.83
- Reduce momentum in Hartree term (momentum in BZ + reciprocal lattice): average rubric score 1.83
- Reduce momentum in Fock term (momentum in BZ + reciprocal lattice): average rubric score 2.00
- Combine the Hartree and Fock term: average rubric score 2.00

## Category Means

- final_answer_accuracy: 1.75
- follow_instructions: 1.88
- in_paper: 1.50
- math_derivation: 1.88
- physics_logic: 2.00
- prompt_quality: 1.81

## High-Signal Placeholder Mismatches

- Construct Kinetic Hamiltonian (continuum version, single-particle) | real|momentum | LLM=`momentum` | reference=`real`
- Construct Kinetic Hamiltonian (continuum version, single-particle) | single-particle|second-quantized | LLM=`second-quantized` | reference=`single-particle`
- Construct Kinetic Hamiltonian (continuum version, single-particle) | degrees_of_freedom | LLM=`valleys, layers, and momentum` | reference=`valley index (+K and -K valley), layer index (top and bottom layer).`
- Construct Kinetic Hamiltonian (continuum version, single-particle) | order | LLM=`bottom layer and top layer` | reference=`(+K,bottom), (+K,top), (-K,bottom), (-K,top)`
- Define each term in Kinetic Hamiltonian (continuum version) | electrons|holes | LLM=`electrons` | reference=`holes`
- Define each term in Kinetic Hamiltonian (continuum version) | momentum_shift | LLM=`$\bm{\kappa}=\frac{4\pi}{3a_M}\left(1,0\right)$` | reference=`$+\kappa$ and $-\kappa$`
- Define each term in Kinetic Hamiltonian (continuum version) | shifted_Ek | LLM=`$\bm{k}-\tau \bm{\kappa}$` | reference=`$E_{t,+K}$ and $E_{t,-K}$`
- Define each term in Kinetic Hamiltonian (continuum version) | expression_kinetic | LLM=`` | reference=`$H_{Kinetic}(r) = \begin{pmatrix} E_{+K,b} & 0 & 0 & 0 \\ 0 & E_{+K,t} & 0 & 0 \\ 0 & 0 & E_{-K,b} & 0 \\ 0 & 0 & 0 & E_{-K,t} \end{pmatrix}$`
- Construct Potential Hamiltonian (continuum version) | real|momentum | LLM=`momentum` | reference=`real`
- Construct Potential Hamiltonian (continuum version) | single-particle|second-quantized | LLM=`second-quantized` | reference=`single-particle`
- Construct Potential Hamiltonian (continuum version) | diagonal_potential | LLM=`$-\frac{\hbar^2\bm{k}^2}{2m_\mathfrak{b}}+\Delta_{\mathfrak{b}}(\bm{r})$ and $-\frac{\hbar^2\left(\bm{k}-\tau \bm{\kappa}\right)^2}{2m_\mathfrak{t}}+ \Delta_\mathfrak{t}(\bm{r})+V_{z\mathfrak{t}}$` | reference=`$\Delta_l(r)$`
- Construct Potential Hamiltonian (continuum version) | potential_degrees_of_freedom | LLM=`the bottom layer ($\mathfrak{b}$) and top layer ($\mathfrak{t}$)` | reference=`$t$ and $b$`
- Define each term in Potential Hamiltonian (continuum version) | Potential_variables | LLM=`$\Delta_{\mathfrak{b}}(\bm{r})$, $\Delta_{\text{T},\tau}(\bm{r})$, $-\frac{\hbar^2\bm{k}^2}{2m_\mathfrak{b}}$, $-\frac{\hbar^2\left(\bm{k}-\tau \bm{\kappa}\right)^2}{2m_\mathfrak{t}}$, and $V_{z\mathfrak{t}}$` | reference=`the intralayer potential $\Delta_l(r)$ and interlayer tunneling $\Delta_{T,\tau}(r)$`
- Define each term in Potential Hamiltonian (continuum version) | expression_diag | LLM=`$-\frac{\hbar^2\bm{k}^2}{2m_\mathfrak{b}}+\Delta_{\mathfrak{b}}(\bm{r})$ and $-\frac{\hbar^2\left(\bm{k}-\tau \bm{\kappa}\right)^2}{2m_\mathfrak{t}}+ \Delta_\mathfrak{t}(\bm{r})+V_{z\mathfrak{t}}$` | reference=`$\Delta_t(r) = V_{zt}$ and $\Delta_b(r)$ is $2V_b\sum_{j=1,3,5} \cos(g_j \cdot r  +\psi_b)$`
- Define each term in Potential Hamiltonian (continuum version) | expression_offdiag | LLM=`$\Delta_{\text{T},\tau}(\bm{r})$ and $\Delta_{\text{T},\tau}^\dag(\bm{r})$` | reference=`$\Delta_{T,+K}(r)=w(1+\omega^{1}e^{i g_2 \cdot r} + \omega^{2} e^{i g_3 \cdot r})$ and $\Delta_{T,-K}(r)=-w(1+\omega^{-1}e^{-i g_2 \cdot r} + \omega^{-2} e^{-i g_3 \cdot r})$.`
- Define each term in Potential Hamiltonian (continuum version) | expression_Potential | LLM=`` | reference=`$H_{Potential}(r) = \begin{pmatrix} \Delta_b(r) & \Delta_{T,+K}(r) & 0 & 0 \\ \Delta_{T,+K}^*(r) & \Delta_t(r) & 0 & 0 \\ 0 & 0 & \Delta_b(r) & \Delta_{T,-K}(r) \\ 0 & 0 & \Delta_{T,-K}^*(r) & \Delta_t(r) \end{pmatrix}$`
- Convert from single-particle to second-quantized form, return in matrix | second_nonint_symbol | LLM=`` | reference=`$\hat{H}^{0}$`
- Convert from single-particle to second-quantized form, return in summation (expand the matrix) | expression_second_nonint | LLM=`` | reference=`$\hat{H}^{0} = \int dr \vec{\psi}^\dagger(r) H^{0}(r) \vec{\psi}(r)$`
- Convert noninteracting Hamiltonian in real space to momentum space (continuum version) | definition_of_Fourier_Transformation | LLM=`$\hat{\mathcal{H}}_0=\sum_{\bm{k}_{\alpha},\bm{k}_{\beta}}\sum_{l_{\alpha},l_{\beta}}\sum_{\tau} h_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}^{(\tau)} c_{\bm{k}_{\alpha},l_{\alpha},\tau}^\dagger c_{\bm{k}_{\beta},l_{\beta},\tau}$` | reference=`$c_{\tau,l}^\dagger(k)= \frac{1}{\sqrt{V}} \int dr \psi_{\tau,l}^\dagger(r) e^{i k \cdot r}$`
- Particle-hole transformation | expression_particle_Ham | LLM=`` | reference=`$\hat{H}^{0} = \sum_{\tau, l_1, l_2, k_1, k_2} c_{\tau,l_1}^\dagger(k_1) H^{0}_{\tau,l_1,\tau,l_2}(k_1,k_2) c_{\tau,l_2}(k_2)$`

## Continuum Hamiltonian

`\begin{equation}\label{eq:Ham} H_{\tau}=\begin{pmatrix} -\frac{\hbar^2\bm{k}^2}{2m_\mathfrak{b}}+\Delta_{\mathfrak{b}}(\bm{r}) & \Delta_{\text{T},\tau}(\bm{r})\\ \Delta_{\text{T},\tau}^\dag(\bm{r}) & -\frac{\hbar^2\left(\bm{k}-\tau \bm{\kappa}\right)^2}{2m_\mathfrak{t}}+ \Delta_\mathfrak{t}(\bm{r})+V_{z\mathfrak{t}} \end{pmatrix}, \end{equation}`

## Full Interacting Hamiltonian

`\begin{equation}\label{eq:full} \begin{split} \hat{\mathcal{H}}&=\hat{\mathcal{H}}_1+\hat{\mathcal{H}}_{\text{int}},\\ \hat{\mathcal{H}}_1&=\sum_{\bm{k}_{\alpha},\bm{k}_{\beta}}\sum_{l_{\alpha},l_{\beta}}\sum_{\tau} \tilde{h}^{(\tau)}_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}} b_{\bm{k}_{\alpha},l_{\alpha},\tau}^\dagger b_{\bm{k}_{\beta},l_{\beta},\tau},\\ \hat{\mathcal{H}}_{\text{int}}&=\frac{1}{2A} \sum_{\bm{k}_{\alpha},\bm{k}_{\beta},\bm{k}_{\gamma},\bm{k}_{\delta} }\sum_{l_{\alpha},l_{\beta}}\sum_{\tau_{\alpha},\tau_{\beta}} V(\bm{k}_{\alpha}-\bm{k}_{\delta}) b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \delta_{\bm{k}_{\alpha}+\bm{k}_{\beta}, \bm{k}_{\delta}+\bm{k}_{\gamma}}, \end{split} \end{equation}`

## Related Work Titles

- paper_000.pdf: MIT Open Access Articles
- paper_001.pdf: Language Models are Few-Shot Learners
- paper_002.pdf: Solving Quantitative Reasoning Problems with
- paper_003.pdf: Galactica: A Large Language Model for Science
- paper_004.pdf: Training Compute-Optimal Large Language Models