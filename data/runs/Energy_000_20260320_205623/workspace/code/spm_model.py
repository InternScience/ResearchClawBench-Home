"""
02_spm_model.py - Single Particle Model (SPM) for Li-ion battery simulation.
Implements the electrochemical-aging-thermal (ECAT) coupled model.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Physical constants
F = 96485.0       # Faraday constant (C/mol)
R_gas = 8.314     # Universal gas constant (J/(mol·K))

class SPMParameters:
    """Default parameters for an 18650 NCM/graphite cell (2 Ah nominal)."""
    def __init__(self, **kwargs):
        # Geometry
        self.Rp_pos = 5.0e-6       # Positive particle radius (m)
        self.Rp_neg = 10.0e-6      # Negative particle radius (m)
        self.L_pos = 70.0e-6       # Positive electrode thickness (m)
        self.L_neg = 73.0e-6       # Negative electrode thickness (m)
        self.L_sep = 25.0e-6       # Separator thickness (m)
        self.A_cell = 0.1          # Electrode area (m^2)

        # Active material volume fractions
        self.eps_s_pos = 0.50
        self.eps_s_neg = 0.58

        # Maximum Li concentration (mol/m^3)
        self.cs_max_pos = 51385.0
        self.cs_max_neg = 30555.0

        # Solid-phase diffusion coefficients (m^2/s)
        self.Ds_pos = 1.0e-14
        self.Ds_neg = 3.9e-14

        # Reaction rate constants (m^2.5 / (mol^0.5 · s))
        self.k_pos = 2.0e-11
        self.k_neg = 2.0e-11

        # Electrolyte concentration (mol/m^3)
        self.ce0 = 1000.0

        # Initial stoichiometries
        self.x0 = 0.5        # Positive (charged state)
        self.y0 = 0.8         # Negative (charged state)
        self.x100 = 0.99      # Positive (discharged state)
        self.y100 = 0.01      # Negative (discharged state)

        # SEI parameters (aging)
        self.R_SEI_0 = 0.01       # Initial SEI resistance (Ohm·m^2)
        self.k_SEI = 1.0e-12      # SEI growth rate constant (m/s)
        self.D_SEI = 1.0e-20      # Solvent diffusivity in SEI (m^2/s)
        self.M_SEI = 0.162        # SEI molar mass (kg/mol)
        self.rho_SEI = 1690.0     # SEI density (kg/m^3)
        self.delta_SEI_0 = 5.0e-9 # Initial SEI thickness (m)

        # Thermal parameters
        self.Cp = 750.0       # Specific heat capacity (J/(kg·K))
        self.m_cell = 0.045   # Cell mass (kg)
        self.h_conv = 5.0     # Convective heat transfer coeff (W/(m^2·K))
        self.A_surf = 0.004   # Cell surface area (m^2)
        self.T_amb = 298.15   # Ambient temperature (K)

        # Transfer coefficients
        self.alpha_a = 0.5
        self.alpha_c = 0.5

        # Contact resistance
        self.R_contact = 0.02  # Ohm

        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def ocp_positive(x):
    """Open circuit potential for NCM positive electrode as function of stoichiometry x."""
    # Polynomial fit based on literature for NCM (Li_x Ni_0.33 Mn_0.33 Co_0.33 O_2)
    U = (4.3452 - 1.8607 * x + 1.4051 * x**2 - 0.8921 * x**3
         + 2.4577 * (np.exp(-40.0 * (x - 0.49))
         - 0.0275 * (1.0 / (0.998 - x)**0.5 - 1.0 / (0.998 - 0.5)**0.5)))
    return np.clip(U, 2.5, 4.4)


def ocp_negative(y):
    """Open circuit potential for graphite negative electrode."""
    # Polynomial fit for graphite (Li_y C_6)
    U = (0.7222 + 0.1387 * y + 0.029 * y**0.5
         - 0.0172 / y + 0.0019 / y**1.5
         + 0.2808 * np.exp(0.90 - 15.0 * y)
         - 0.7984 * np.exp(0.4465 * y - 0.4108))
    return np.clip(U, 0.0, 1.5)


class SPMSimulator:
    """
    Single Particle Model with aging and thermal coupling.
    Uses volume-averaged approach with Fick's law in spherical coordinates.
    """
    def __init__(self, params=None):
        self.p = params if params else SPMParameters()

    def _calc_specific_interfacial_area(self, electrode='pos'):
        """a_s = 3 * eps_s / R_p"""
        if electrode == 'pos':
            return 3.0 * self.p.eps_s_pos / self.p.Rp_pos
        else:
            return 3.0 * self.p.eps_s_neg / self.p.Rp_neg

    def _exchange_current_density(self, cs_surf, cs_max, k, ce, T):
        """Butler-Volmer exchange current density."""
        # j0 = F * k * ce^0.5 * cs_surf^0.5 * (cs_max - cs_surf)^0.5
        cs_surf = np.clip(cs_surf, 1e-3, cs_max - 1e-3)
        return F * k * (ce**0.5) * (cs_surf**0.5) * ((cs_max - cs_surf)**0.5)

    def simulate_discharge(self, I_app, t_final, dt=1.0, V_cutoff=2.5, n_r=10):
        """
        Simulate constant-current discharge.

        Args:
            I_app: Applied current (A, positive = discharge)
            t_final: Maximum simulation time (s)
            dt: Time step (s)
            V_cutoff: Cutoff voltage (V)
            n_r: Number of radial discretization points

        Returns:
            dict with time, voltage, temperature, capacity, SOC arrays
        """
        p = self.p

        # Radial discretization for both electrodes
        r_pos = np.linspace(0, p.Rp_pos, n_r)
        r_neg = np.linspace(0, p.Rp_neg, n_r)
        dr_pos = p.Rp_pos / (n_r - 1)
        dr_neg = p.Rp_neg / (n_r - 1)

        # Initial concentration profiles (uniform)
        cs_pos = np.full(n_r, p.x0 * p.cs_max_pos)
        cs_neg = np.full(n_r, p.y0 * p.cs_max_neg)

        # Initial state
        T = p.T_amb
        delta_SEI = p.delta_SEI_0
        R_SEI = p.R_SEI_0

        # Interfacial areas
        a_s_pos = self._calc_specific_interfacial_area('pos')
        a_s_neg = self._calc_specific_interfacial_area('neg')

        # Current density on electrode surface
        j_pos = I_app / (a_s_pos * p.L_pos * p.A_cell * F)
        j_neg = -I_app / (a_s_neg * p.L_neg * p.A_cell * F)

        # Storage
        times = []
        voltages = []
        temperatures = []
        capacities = []

        t = 0.0
        Q = 0.0  # Accumulated charge (Ah)

        while t <= t_final:
            # Surface concentrations
            cs_surf_pos = cs_pos[-1]
            cs_surf_neg = cs_neg[-1]

            # Stoichiometries
            x = cs_surf_pos / p.cs_max_pos
            y = cs_surf_neg / p.cs_max_neg

            # OCP
            U_pos = ocp_positive(np.clip(x, 0.01, 0.99))
            U_neg = ocp_negative(np.clip(y, 0.01, 0.99))

            # Exchange current densities
            j0_pos = self._exchange_current_density(
                cs_surf_pos, p.cs_max_pos, p.k_pos, p.ce0, T)
            j0_neg = self._exchange_current_density(
                cs_surf_neg, p.cs_max_neg, p.k_neg, p.ce0, T)

            # Overpotentials (simplified inverse Butler-Volmer)
            # eta = (RT/F) * arcsinh(j_n / (2*j0))
            eta_pos = (R_gas * T / F) * np.arcsinh(j_pos * F / (2.0 * j0_pos + 1e-20))
            eta_neg = (R_gas * T / F) * np.arcsinh(j_neg * F / (2.0 * j0_neg + 1e-20))

            # Terminal voltage
            V = U_pos - U_neg + eta_pos - eta_neg - I_app * (R_SEI / p.A_cell + p.R_contact)

            # Store
            times.append(t)
            voltages.append(float(V))
            temperatures.append(float(T - 273.15))
            capacities.append(Q)

            if V < V_cutoff and t > 10:
                break

            # ---- Update solid-phase diffusion (Fick's law, spherical) ----
            # Finite difference in radial direction
            cs_pos_new = cs_pos.copy()
            cs_neg_new = cs_neg.copy()

            for i in range(1, n_r - 1):
                # d(cs)/dt = Ds * (d2cs/dr2 + 2/r * dcs/dr)
                d2cs_pos = (cs_pos[i+1] - 2*cs_pos[i] + cs_pos[i-1]) / dr_pos**2
                dcs_pos = (cs_pos[i+1] - cs_pos[i-1]) / (2*dr_pos)
                cs_pos_new[i] = cs_pos[i] + dt * p.Ds_pos * (d2cs_pos + 2.0/(r_pos[i]+1e-20) * dcs_pos)

                d2cs_neg = (cs_neg[i+1] - 2*cs_neg[i] + cs_neg[i-1]) / dr_neg**2
                dcs_neg = (cs_neg[i+1] - cs_neg[i-1]) / (2*dr_neg)
                cs_neg_new[i] = cs_neg[i] + dt * p.Ds_neg * (d2cs_neg + 2.0/(r_neg[i]+1e-20) * dcs_neg)

            # Boundary conditions
            # r=0: symmetry (dcs/dr = 0)
            cs_pos_new[0] = cs_pos_new[1]
            cs_neg_new[0] = cs_neg_new[1]

            # r=R: flux = -j_n (pore wall flux)
            cs_pos_new[-1] = cs_pos[-1] - dt * j_pos / (p.Ds_pos + 1e-30) * dr_pos
            cs_neg_new[-1] = cs_neg[-1] - dt * j_neg / (p.Ds_neg + 1e-30) * dr_neg

            # Clip to physical bounds
            cs_pos = np.clip(cs_pos_new, 1.0, p.cs_max_pos - 1.0)
            cs_neg = np.clip(cs_neg_new, 1.0, p.cs_max_neg - 1.0)

            # ---- Thermal model ----
            # Q_rxn = I * (V_OCV - V)  (reaction heat)
            V_ocv = U_pos - U_neg
            Q_rxn = I_app * (V_ocv - V)
            # Q_rev ~ I * T * dU/dT (reversible heat, simplified)
            Q_rev = I_app * T * 0.0001  # small entropic term
            # Cooling
            Q_cool = p.h_conv * p.A_surf * (T - p.T_amb)
            # Energy balance: m*Cp*dT/dt = Q_rxn + Q_rev - Q_cool
            dTdt = (Q_rxn + Q_rev - Q_cool) / (p.m_cell * p.Cp)
            T = T + dt * dTdt

            # ---- SEI growth (simplified diffusion-limited) ----
            # d(delta)/dt = k_SEI * exp(-delta/D_SEI_scale)
            d_delta = p.k_SEI * np.exp(-delta_SEI / (p.D_SEI * 1e9 + 1e-30)) * dt
            delta_SEI = delta_SEI + d_delta
            R_SEI = p.R_SEI_0 + delta_SEI / (5e-6 + 1e-30)  # kappa_SEI ~ 5e-6 S/m

            # Update capacity
            Q += I_app * dt / 3600.0

            t += dt

        return {
            'time': np.array(times),
            'voltage': np.array(voltages),
            'temperature': np.array(temperatures),
            'capacity': np.array(capacities),
        }


# ============================================================
# Parameter space definition for identification
# ============================================================
PARAM_BOUNDS = {
    # key: (lower, upper, log_scale, description)
    'Rp_pos':      (1e-6, 15e-6, True,  'Positive particle radius (m)'),
    'Rp_neg':      (3e-6, 20e-6, True,  'Negative particle radius (m)'),
    'Ds_pos':      (1e-16, 1e-12, True,  'Positive diffusion coeff (m^2/s)'),
    'Ds_neg':      (1e-16, 1e-12, True,  'Negative diffusion coeff (m^2/s)'),
    'k_pos':       (1e-13, 1e-9, True,   'Positive reaction rate (m^2.5/(mol^0.5·s))'),
    'k_neg':       (1e-13, 1e-9, True,   'Negative reaction rate'),
    'eps_s_pos':   (0.3, 0.7, False,     'Positive volume fraction'),
    'eps_s_neg':   (0.4, 0.75, False,    'Negative volume fraction'),
    'R_SEI_0':     (0.001, 0.1, True,    'Initial SEI resistance (Ohm·m^2)'),
    'R_contact':   (0.005, 0.1, True,    'Contact resistance (Ohm)'),
    'h_conv':      (1.0, 20.0, False,    'Convective heat transfer coeff (W/(m^2·K))'),
    'x0':          (0.3, 0.6, False,     'Initial positive stoichiometry'),
    'y0':          (0.7, 0.95, False,    'Initial negative stoichiometry'),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
N_PARAMS = len(PARAM_NAMES)


def params_from_vector(vec):
    """Convert parameter vector to SPMParameters object."""
    kwargs = {}
    for i, name in enumerate(PARAM_NAMES):
        lb, ub, log_scale, _ = PARAM_BOUNDS[name]
        if log_scale:
            val = 10**(np.log10(lb) + vec[i] * (np.log10(ub) - np.log10(lb)))
        else:
            val = lb + vec[i] * (ub - lb)
        kwargs[name] = val
    return SPMParameters(**kwargs)


def vector_from_params(params):
    """Convert SPMParameters to normalized [0,1] vector."""
    vec = np.zeros(N_PARAMS)
    for i, name in enumerate(PARAM_NAMES):
        lb, ub, log_scale, _ = PARAM_BOUNDS[name]
        val = getattr(params, name)
        if log_scale:
            vec[i] = (np.log10(val) - np.log10(lb)) / (np.log10(ub) - np.log10(lb))
        else:
            vec[i] = (val - lb) / (ub - lb)
    return np.clip(vec, 0, 1)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR = os.path.join(WORKSPACE, 'report', 'images')

    # Test simulation with default parameters
    print("Testing SPM simulation...")
    sim = SPMSimulator()

    # Simulate at different C-rates
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('SPM Simulation - Discharge at Different C-rates', fontweight='bold')

    nominal_cap = 2.0  # Ah
    for c_rate, color in [(0.5, 'blue'), (1.0, 'green'), (2.0, 'red')]:
        I_app = c_rate * nominal_cap
        t_max = 3600.0 / c_rate * 1.2
        result = sim.simulate_discharge(I_app, t_max, dt=1.0)

        axes[0].plot(result['capacity'], result['voltage'], color=color,
                    label=f'{c_rate}C')
        axes[1].plot(result['time']/60, result['temperature'], color=color,
                    label=f'{c_rate}C')
        axes[2].plot(result['time']/60, result['voltage'], color=color,
                    label=f'{c_rate}C')

    axes[0].set_xlabel('Capacity (Ah)'); axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('Voltage vs Capacity'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time (min)'); axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Temperature'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (min)'); axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('Voltage vs Time'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'spm_baseline.png'))
    plt.close()
    print("Saved spm_baseline.png")
    print(f"1C discharge: {result['capacity'][-1]:.3f} Ah, time={result['time'][-1]:.0f}s")
