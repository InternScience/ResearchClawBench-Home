"""Visualization utilities for VOS framework results"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_convergence_comparison(results, save_path='report/images/convergence_comparison.png'):
    """Plot convergence comparison of all methods"""
    
    f_star = results['f_star']
    problem = results['problem']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract histories
    gd_obj = np.array(results['gradient_descent']['history']['objective'])
    nest_obj = np.array(results['nesterov']['history']['objective'])
    admm_obj = np.array(results['admm']['history']['objective'])
    
    # Plot 1: Objective value vs iteration
    ax = axes[0, 0]
    ax.plot(gd_obj, label='Gradient Descent (ISTA)', linewidth=2)
    ax.plot(nest_obj, label='Nesterov (FISTA)', linewidth=2)
    ax.plot(admm_obj, label='ADMM', linewidth=2)
    ax.axhline(y=f_star, color='k', linestyle='--', label='f* (estimated)', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Value Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Suboptimality (log scale)
    ax = axes[0, 1]
    gd_gap = np.maximum(gd_obj - f_star, 1e-16)
    nest_gap = np.maximum(nest_obj - f_star, 1e-16)
    admm_gap = np.maximum(admm_obj - f_star, 1e-16)
    
    ax.semilogy(gd_gap, label='Gradient Descent', linewidth=2)
    ax.semilogy(nest_gap, label='Nesterov', linewidth=2)
    ax.semilogy(admm_gap, label='ADMM', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x_k) - f* (log scale)')
    ax.set_title('Convergence Rate Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Linear convergence validation (log-log)
    ax = axes[1, 0]
    iterations = np.arange(len(gd_gap))
    
    # Fit lines to verify linear convergence
    gd_fit = np.polyfit(iterations[50:150], np.log(gd_gap[50:150]), 1)
    nest_fit = np.polyfit(iterations[50:150], np.log(nest_gap[50:150]), 1)
    
    ax.semilogy(iterations, gd_gap, 'b-', alpha=0.5, label=f'GD (slope={gd_fit[0]:.4f})')
    ax.semilogy(iterations, nest_gap, 'r-', alpha=0.5, label=f'Nesterov (slope={nest_fit[0]:.4f})')
    ax.semilogy(iterations, admm_gap, 'g-', alpha=0.5, label='ADMM')
    
    # Theoretical slopes
    kappa = problem.L / problem.mu
    ax.axhline(y=gd_gap[0] * np.exp(-1/kappa * iterations), color='b', 
               linestyle='--', alpha=0.5, label='GD theory')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Suboptimality (log scale)')
    ax.set_title('Linear Convergence Validation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence rate summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute final errors
    gd_final = gd_obj[-1] - f_star
    nest_final = nest_obj[-1] - f_star
    admm_final = admm_obj[-1] - f_star
    
    # Count iterations to reach tolerance
    tol = 1e-3
    gd_iters = np.where(gd_gap < tol)[0]
    nest_iters = np.where(nest_gap < tol)[0]
    admm_iters = np.where(admm_gap < tol)[0]
    
    gd_iter_count = gd_iters[0] if len(gd_iters) > 0 else len(gd_gap)
    nest_iter_count = nest_iters[0] if len(nest_iters) > 0 else len(nest_gap)
    admm_iter_count = admm_iters[0] if len(admm_iters) > 0 else len(admm_gap)
    
    summary_text = f"""
    Convergence Summary:
    
    Condition Number κ = {kappa:.1e}
    
    Final Suboptimality (f(x) - f*):
    • Gradient Descent: {gd_final:.6e}
    • Nesterov (FISTA): {nest_final:.6e}
    • ADMM: {admm_final:.6e}
    
    Iterations to reach {tol:.0e} tolerance:
    • Gradient Descent: {gd_iter_count}
    • Nesterov (FISTA): {nest_iter_count}
    • ADMM: {admm_iter_count}
    
    Observed Convergence Rates:
    • GD: O((1-μ/L)^k) ≈ linear
    • Nesterov: O(1/k²) then linear
    • ADMM: O(1/k) then linear
    
    Speedup of Nesterov over GD:
    {gd_iter_count/max(nest_iter_count,1):.2f}x faster
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence comparison saved to {save_path}")
    return fig


def plot_lyapunov_analysis(results, save_path='report/images/lyapunov_analysis.png'):
    """Plot Lyapunov function analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract Lyapunov histories
    gd_lyap = np.array(results['gradient_descent']['history']['lyapunov'])
    nest_lyap = np.array(results['nesterov']['history']['lyapunov'])
    admm_lyap = np.array(results['admm']['history']['lyapunov'])
    
    # Plot 1: Lyapunov decay
    ax = axes[0, 0]
    ax.semilogy(gd_lyap, label='Gradient Descent', linewidth=2)
    ax.semilogy(nest_lyap, label='Nesterov', linewidth=2)
    ax.semilogy(admm_lyap, label='ADMM', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Lyapunov Function (log scale)')
    ax.set_title('Lyapunov Function Decay (Stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Normalized Lyapunov (showing monotonicity)
    ax = axes[0, 1]
    if len(gd_lyap) > 1:
        ax.plot(np.diff(gd_lyap), 'b-', alpha=0.7, label='GD ΔV')
    if len(nest_lyap) > 1:
        ax.plot(np.diff(nest_lyap), 'r-', alpha=0.7, label='Nesterov ΔV')
    if len(admm_lyap) > 1:
        ax.plot(np.diff(admm_lyap), 'g-', alpha=0.7, label='ADMM ΔV')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ΔV = V_{k+1} - V_k')
    ax.set_title('Lyapunov Decrement (Monotonicity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Continuous-time Lyapunov (VOS framework)
    ax = axes[1, 0]
    if 'vos_continuous' in results:
        t_cont = results['vos_continuous']['t']
        energy = results['vos_continuous']['energy']
        ax.plot(t_cont, energy, 'purple', linewidth=2)
        ax.set_xlabel('Time t')
        ax.set_ylabel('Energy E(t)')
        ax.set_title('Continuous-Time Lyapunov (VOS Framework)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Lyapunov theory
    ax = axes[1, 1]
    ax.axis('off')
    theory_text = r"""
    Lyapunov Function Analysis:
    
    For Gradient Descent:
    V_k = f(x_k) + (1/2s)||x_k - x_{k-1}||²
    
    For Nesterov's Method:
    V_k = t_k²(f(x_k) - f*) + 2||z_k - x*||²
    where z_k = x_{k-1} + t_k(x_k - x_{k-1})
    
    For ADMM:
    V_k = L_ρ(x_k, z_k, u_k) + (ρ/2)||u_k||²
    
    Continuous (VOS):
    E(t) = t²(f(X) - f*) + 2||Ẋ||²
    
    Key Property: V_{k+1} ≤ V_k (monotonic)
    This guarantees convergence to equilibrium.
    
    The decay rate provides convergence rate bounds.
    """
    ax.text(0.1, 0.5, theory_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Lyapunov analysis saved to {save_path}")
    return fig


def plot_phase_space(results, save_path='report/images/phase_space.png'):
    """Plot phase space trajectories for 2D projection"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Project to 2D using first two principal components or just first two coordinates
    def project_2d(trajectory):
        """Project trajectory to 2D"""
        if trajectory.shape[1] >= 2:
            return trajectory[:, 0], trajectory[:, 1]
        else:
            return np.arange(len(trajectory)), trajectory[:, 0]
    
    # Get iterates
    gd_iters = np.array(results['gradient_descent']['history']['iterates'])
    nest_iters = np.array(results['nesterov']['history']['iterates'])
    admm_primal = []  # ADMM doesn't store full trajectory
    
    # Plot 1: GD trajectory in 2D
    ax = axes[0, 0]
    if len(gd_iters) > 0:
        x1, x2 = project_2d(gd_iters)
        scatter = ax.scatter(x1, x2, c=np.arange(len(x1)), cmap='viridis', s=10)
        ax.plot(x1, x2, 'b-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title('Gradient Descent Trajectory')
        plt.colorbar(scatter, ax=ax, label='Iteration')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Nesterov trajectory
    ax = axes[0, 1]
    if len(nest_iters) > 0:
        x1, x2 = project_2d(nest_iters)
        scatter = ax.scatter(x1, x2, c=np.arange(len(x1)), cmap='plasma', s=10)
        ax.plot(x1, x2, 'r-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title('Nesterov Accelerated Trajectory')
        plt.colorbar(scatter, ax=ax, label='Iteration')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Continuous phase space
    ax = axes[1, 0]
    if 'vos_continuous' in results:
        X_cont = results['vos_continuous']['X']
        V_cont = results['vos_continuous']['V']
        
        # Plot position vs velocity for first coordinate
        ax.plot(X_cont[:, 0], V_cont[:, 0], 'purple', linewidth=1)
        ax.set_xlabel('Position X₁(t)')
        ax.set_ylabel('Velocity Ẋ₁(t)')
        ax.set_title('Continuous Phase Space (VOS)')
        ax.grid(True, alpha=0.3)
        
        # Mark direction
        n_arrows = 10
        indices = np.linspace(0, len(X_cont)-10, n_arrows, dtype=int)
        for i in indices:
            dx = X_cont[i+5, 0] - X_cont[i, 0]
            dy = V_cont[i+5, 0] - V_cont[i, 0]
            ax.annotate('', xy=(X_cont[i+5, 0], V_cont[i+5, 0]),
                       xytext=(X_cont[i, 0], V_cont[i, 0]),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    
    # Plot 4: Theory explanation
    ax = axes[1, 1]
    ax.axis('off')
    theory_text = r"""
    Phase Space Analysis:
    
    Gradient Descent:
    • Direct path to optimum
    • No momentum (overdamped)
    
    Nesterov's Method:
    • Momentum causes oscillation
    • Underdamped dynamics
    • Overshoots before settling
    
    VOS ODE: Ẍ + (3/t)Ẋ + ∇f(X) = 0
    • Damping decreases over time
    • Starts overdamped (3/t large)
    • Becomes underdamped (3/t small)
    
    This explains Nesterov's
    oscillatory behavior!
    """
    ax.text(0.1, 0.5, theory_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Phase space plot saved to {save_path}")
    return fig


def plot_linear_convergence(results, save_path='report/images/linear_convergence.png'):
    """Plot validation of linear convergence rates"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    f_star = results['f_star']
    problem = results['problem']
    
    gd_obj = np.array(results['gradient_descent']['history']['objective'])
    nest_obj = np.array(results['nesterov']['history']['objective'])
    admm_obj = np.array(results['admm']['history']['objective'])
    
    gd_gap = np.maximum(gd_obj - f_star, 1e-16)
    nest_gap = np.maximum(nest_obj - f_star, 1e-16)
    admm_gap = np.maximum(admm_obj - f_star, 1e-16)
    
    iterations = np.arange(len(gd_gap))
    
    # Plot 1: Linear convergence on log scale
    ax = axes[0, 0]
    ax.semilogy(iterations, gd_gap, 'b-', linewidth=2, label='GD')
    ax.semilogy(iterations[:len(nest_gap)], nest_gap, 'r-', linewidth=2, label='Nesterov')
    ax.semilogy(iterations[:len(admm_gap)], admm_gap, 'g-', linewidth=2, label='ADMM')
    
    # Fit exponential decay
    from scipy.optimize import curve_fit
    def exp_decay(k, a, r):
        return a * np.exp(-r * k)
    
    try:
        # Fit on middle section
        mid_start, mid_end = 50, min(200, len(gd_gap)-1)
        popt_gd, _ = curve_fit(exp_decay, iterations[mid_start:mid_end], 
                               gd_gap[mid_start:mid_end], p0=[gd_gap[0], 0.01])
        ax.semilogy(iterations, exp_decay(iterations, *popt_gd), 'b--', 
                   label=f'GD fit: exp(-{popt_gd[1]:.4f}k)')
    except:
        pass
    
    ax.set_xlabel('Iteration k')
    ax.set_ylabel('f(x_k) - f* (log scale)')
    ax.set_title('Linear Convergence: Exponential Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Contraction factor
    ax = axes[0, 1]
    gd_ratio = gd_gap[1:] / gd_gap[:-1]
    nest_ratio = nest_gap[1:] / nest_gap[:-1]
    admm_ratio = admm_gap[1:] / admm_gap[:-1]
    
    ax.plot(gd_ratio, 'b-', alpha=0.7, label='GD')
    ax.plot(nest_ratio, 'r-', alpha=0.7, label='Nesterov')
    ax.plot(admm_ratio, 'g-', alpha=0.7, label='ADMM')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('(f(x_{k+1}) - f*) / (f(x_k) - f*)')
    ax.set_title('Contraction Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Theoretical vs empirical rate
    ax = axes[1, 0]
    kappa = problem.L / problem.mu
    
    # Theoretical rates
    rho_gd_theory = 1 - problem.mu / problem.L
    rho_nest_theory = 1 - np.sqrt(problem.mu / problem.L)
    
    # Empirical rates (average of last 100 iterations)
    n_avg = min(100, len(gd_ratio))
    rho_gd_emp = np.mean(gd_ratio[-n_avg:]) if len(gd_ratio) >= n_avg else np.nan
    rho_nest_emp = np.mean(nest_ratio[-n_avg:]) if len(nest_ratio) >= n_avg else np.nan
    
    methods = ['GD', 'Nesterov']
    theory_rates = [rho_gd_theory, rho_nest_theory]
    emp_rates = [rho_gd_emp, rho_nest_emp]
    
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, theory_rates, width, label='Theory', alpha=0.8)
    ax.bar(x + width/2, emp_rates, width, label='Empirical', alpha=0.8)
    ax.set_ylabel('Contraction Factor ρ')
    ax.set_title(f'Linear Convergence Rates (κ={kappa:.1e})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Linear Convergence Analysis:
    
    Problem Parameters:
    • L (smoothness): {problem.L:.4f}
    • μ (strong convexity): {problem.mu:.6f}
    • Condition number κ: {kappa:.2e}
    
    Theoretical Contraction Factors:
    • GD: ρ = 1 - μ/L = 1 - 1/κ
      = {rho_gd_theory:.6f}
    • Nesterov: ρ = 1 - √(μ/L) = 1 - 1/√κ
      = {rho_nest_theory:.6f}
    
    Observed Contraction:
    • GD: ρ ≈ {rho_gd_emp:.6f}
    • Nesterov: ρ ≈ {rho_nest_emp:.6f}
    
    Key Finding:
    Nesterov achieves O(√κ) acceleration
    over gradient descent!
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Linear convergence plot saved to {save_path}")
    return fig


def create_comparison_table(results, save_path='outputs/comparison_table.csv'):
    """Create quantitative comparison table"""
    
    f_star = results['f_star']
    problem = results['problem']
    
    # Extract final values
    gd_obj = results['gradient_descent']['history']['objective']
    nest_obj = results['nesterov']['history']['objective']
    admm_obj = results['admm']['history']['objective']
    
    # Compute statistics
    def compute_stats(obj_hist):
        final_obj = obj_hist[-1]
        final_gap = final_obj - f_star
        iters_to_1e3 = next((i for i, v in enumerate(obj_hist) if v - f_star < 1e-3), len(obj_hist))
        iters_to_1e6 = next((i for i, v in enumerate(obj_hist) if v - f_star < 1e-6), len(obj_hist))
        return final_obj, final_gap, iters_to_1e3, iters_to_1e6
    
    gd_stats = compute_stats(gd_obj)
    nest_stats = compute_stats(nest_obj)
    admm_stats = compute_stats(admm_obj)
    
    # Create table
    import pandas as pd
    table = pd.DataFrame({
        'Method': ['Gradient Descent (ISTA)', 'Nesterov (FISTA)', 'ADMM'],
        'Final Objective': [gd_stats[0], nest_stats[0], admm_stats[0]],
        'Final Gap (f-f*)': [gd_stats[1], nest_stats[1], admm_stats[1]],
        'Iters to 1e-3': [gd_stats[2], nest_stats[2], admm_stats[2]],
        'Iters to 1e-6': [gd_stats[3], nest_stats[3], admm_stats[3]]
    })
    
    table.to_csv(save_path, index=False)
    print(f"Comparison table saved to {save_path}")
    return table


if __name__ == '__main__':
    print("Visualization utilities loaded")
