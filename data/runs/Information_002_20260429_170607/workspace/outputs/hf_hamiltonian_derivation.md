# Derived Hartree-Fock Hamiltonian for 2111.01152

## 1. Single-particle continuum Hamiltonian
For valley $\tau=\pm1$ and layer basis $(\mathfrak b,\mathfrak t)$, the target paper defines

\[
H_\tau(\mathbf r)=\begin{pmatrix}
-\frac{\hbar^2\mathbf k^2}{2m_\mathfrak b}+\Delta_\mathfrak b(\mathbf r)&\Delta_{T,\tau}(\mathbf r)\\
\Delta_{T,\tau}^{\dagger}(\mathbf r)&-\frac{\hbar^2(\mathbf k-\tau\boldsymbol\kappa)^2}{2m_\mathfrak t}+\Delta_\mathfrak t(\mathbf r)+V_{z\mathfrak t}
\end{pmatrix},
\]
where $\boldsymbol\kappa=4\pi(1,0)/(3a_M)$, $(m_\mathfrak b,m_\mathfrak t)=(0.65,0.35)m_e$, and $\mathbf k=-i\nabla_\mathbf r$ in real space.

The intralayer and tunneling fields are
\[
\Delta_{\mathfrak b}(\mathbf r)=2V_{\mathfrak b}\sum_{j=1,3,5}\cos(\mathbf g_j\cdot\mathbf r+\psi_\mathfrak b),\qquad \Delta_\mathfrak t(\mathbf r)=0,
\]
\[
\Delta_{T,\tau}(\mathbf r)=\tau w\left[1+\omega^\tau e^{i\tau\mathbf g_2\cdot\mathbf r}+\omega^{2\tau}e^{i\tau\mathbf g_3\cdot\mathbf r}\right],\quad \omega=e^{2\pi i/3}.
\]

## 2. Second quantization and momentum representation
The real-space second-quantized noninteracting Hamiltonian is
\[
\hat{\mathcal H}_0=\sum_{\tau}\int d^2r\,\Psi_\tau^\dagger(\mathbf r)H_\tau(\mathbf r)\Psi_\tau(\mathbf r),
\]
with spin inferred by spin-valley-layer locking. Defining
\[
c_{\mathbf k,l,\tau}^{\dagger}=A^{-1/2}\int d^2r\,\psi_{l,\tau}^{\dagger}(\mathbf r)e^{i\mathbf k\cdot\mathbf r},
\]
produces
\[
\hat{\mathcal H}_0=\sum_{\mathbf k_\alpha,\mathbf k_\beta}\sum_{l_\alpha,l_\beta}\sum_\tau h^{(\tau)}_{\mathbf k_\alpha l_\alpha,\mathbf k_\beta l_\beta}c^{\dagger}_{\mathbf k_\alpha,l_\alpha,\tau}c_{\mathbf k_\beta,l_\beta,\tau}.
\]
Bloch periodicity restricts nonzero matrix elements to momentum differences equal to moire reciprocal lattice vectors.

## 3. Hole basis and interaction
Using $b_{\mathbf k,l,\tau}=c^{\dagger}_{\mathbf k,l,\tau}$, normal ordering gives the one-body hole Hamiltonian (dropping constants)
\[
\hat{\mathcal H}_1=\sum_{\mathbf k_\alpha,\mathbf k_\beta}\sum_{l_\alpha,l_\beta}\sum_\tau \tilde h^{(\tau)}_{\mathbf k_\alpha l_\alpha,\mathbf k_\beta l_\beta}b^{\dagger}_{\mathbf k_\alpha,l_\alpha,\tau}b_{\mathbf k_\beta,l_\beta,\tau},\quad \tilde h^{(\tau)}=-[h^{(\tau)}]^T.
\]
The hole-hole interaction is
\[
\hat{\mathcal H}_{\rm int}=\frac{1}{2A}\sum_{\alpha\beta\gamma\delta}\sum_{l_\alpha,l_\beta}\sum_{\tau_\alpha,\tau_\beta}V(\mathbf k_\alpha-\mathbf k_\delta)
 b_\alpha^\dagger b_\beta^\dagger b_\gamma b_\delta\,
\delta_{\mathbf k_\alpha+\mathbf k_\beta,\mathbf k_\delta+\mathbf k_\gamma},
\]
where $b_\alpha=b_{\mathbf k_\alpha,l_\alpha,\tau_\alpha}$, $b_\beta=b_{\mathbf k_\beta,l_\beta,\tau_\beta}$, $b_\gamma=b_{\mathbf k_\gamma,l_\beta,\tau_\beta}$, $b_\delta=b_{\mathbf k_\delta,l_\alpha,\tau_\alpha}$, and
\[
V(\mathbf q)=\frac{2\pi e^2\tanh(|\mathbf q|d)}{\epsilon |\mathbf q|}.
\]

## 4. Hartree-Fock decoupling
Applying Wick's theorem to $b_\alpha^\dagger b_\beta^\dagger b_\gamma b_\delta$ and combining equivalent Hartree and Fock partners cancels the factor $1/2$, giving
\[
\hat{\mathcal H}^{\rm HF}=\hat{\mathcal H}_1+\hat{\mathcal H}^{\rm HF}_{\rm int},
\]
\[
\hat{\mathcal H}^{\rm HF}_{\rm int}=\frac{1}{A}\sum_{\alpha\beta\gamma\delta}\sum_{l_\alpha,l_\beta}\sum_{\tau_\alpha,\tau_\beta} V(\mathbf k_\alpha-\mathbf k_\delta)
\left[\langle b_\alpha^\dagger b_\delta\rangle b_\beta^\dagger b_\gamma-\langle b_\alpha^\dagger b_\gamma\rangle b_\beta^\dagger b_\delta\right]
\delta_{\mathbf k_\alpha+\mathbf k_\beta,\mathbf k_\delta+\mathbf k_\gamma}.
\]
This is the compact source-equation form of the Hartree-Fock Hamiltonian used for validation.
