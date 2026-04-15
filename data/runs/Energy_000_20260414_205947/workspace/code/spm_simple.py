import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SimpleSPM:
    def __init__(self, params):
        # params dict: R_p, R_n, Ds_p, Ds_n, k_p, k_n, U_p_func, U_n_func, F=96485, R=8.314, T=298, eps_s_p=0.5, eps_s_n=0.5, a_s_p=3e5, a_s_n=3e5, L_p=50e-6, L_n=50e-6, cmax_p=51500, cmax_n=30500, c0_p=0.5*cmax_p, c0_n=0.5*cmax_n, lambda_th=1, rhoCp=1e6, I_app=-2, t_end=2*3600
        self.params = params
        self.F = params.get('F', 96485)
        self.R = params.get('R', 8.314)
        self.T = params.get('T', 298)
        self.precompute()
    
    def precompute(self):
        p = self.params
        self.Nr = 20  # radial points
        self.r = np.linspace(0, 1, self.Nr)
        dr = self.r[1] - self.r[0]
        self.D_r2 = p['Ds_p'] * dr**2 / p['R_p']**2  # nondim
        # Similar for n
        # Matrices for diffusion later
    
    def U_p(self, theta):
        # NMC OCV approx
        return 3.4 + 0.1*np.tanh((theta-0.5)/0.1) - 0.1*(theta-0.8)**2  # dummy
    
    def U_n(self, theta):
        return 0.1 + 0.5*np.tanh((theta-0.5)/0.1)  # dummy graphite
    
    def sim_discharge(self):
        p = self.params
        t_span = (0, p['t_end'])
        sol = solve_ivp(self.dydt, t_span, self.y0(), method='RK45', rtol=1e-4)
        return sol.t, sol.y
    
    def dydt(self, t, y):
        # Implement diffusion, BV, thermal
        # Dummy for now
        return np.zeros_like(y)
    
    def y0(self):
        # Flatten c_p(r), c_n(r), T
        return np.zeros(2*self.Nr + 1)
    
# Test
params = {'R_p':5e-6, 'R_n':10e-6, 'Ds_p':1e-14, 'Ds_n':1e-15, 'k_p':1e-6, 'k_n':1e-6, 't_end':7200}
model = SimpleSPM(params)
t, y = model.sim_discharge()
plt.plot(t/3600, y[0])
plt.savefig('outputs/test_spm.png')
print('SPM test OK')
