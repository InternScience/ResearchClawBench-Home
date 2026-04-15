#!/usr/bin/env python3
"""
Main experiment runner for VOS Framework

Runs all optimization algorithms and generates figures for the research report.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

from data_utils import load_lasso_data, plot_data_overview
from vos_framework import (
    LassoProblem, GradientDescent, NesterovAcceleratedGradient, 
    ADMM, VOSFramework, run_experiments
)
from visualizations import (
    plot_convergence_comparison, plot_lyapunov_analysis,
    plot_phase_space, plot_linear_convergence, create_comparison_table
)


def main():
    print("="*60)
    print("Variable and Operator Splitting (VOS) Framework")
    print("Research Experiments")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    data_path = 'data/complex_optimization_data.npy'
    A, b, x_true = load_lasso_data(data_path)
    print(f"    Data loaded: A={A.shape}, b={b.shape}, x_true={x_true.shape}")
    print(f"    True sparsity: {np.count_nonzero(x_true)/len(x_true):.2%}")
    
    # Create data overview plot
    print("\n[2/6] Creating data overview...")
    plot_data_overview(A, b, x_true, 'report/images/data_overview.png')
    
    # Set up problem
    lambda_reg = 0.1
    problem = LassoProblem(A, b, lambda_reg)
    print(f"\n[3/6] Problem setup:")
    print(f"    Lipschitz constant L: {problem.L:.4f}")
    print(f"    Strong convexity μ: {problem.mu:.6f}")
    print(f"    Condition number κ: {problem.L/problem.mu:.2e}")
    print(f"    Regularization λ: {lambda_reg}")
    
    # Run experiments
    print("\n[4/6] Running optimization algorithms...")
    results = run_experiments(data_path, lambda_reg)
    
    # Print results summary
    print("\n[5/6] Results Summary:")
    print(f"    Final GD objective: {results['gradient_descent']['history']['objective'][-1]:.8f}")
    print(f"    Final Nesterov objective: {results['nesterov']['history']['objective'][-1]:.8f}")
    print(f"    Final ADMM objective: {results['admm']['history']['objective'][-1]:.8f}")
    print(f"    Estimated f*: {results['f_star']:.8f}")
    
    # Save convergence results
    print("\n[6/6] Saving results and generating visualizations...")
    np.savez('outputs/convergence_results.npz',
             gd_objective=np.array(results['gradient_descent']['history']['objective']),
             gd_lyapunov=np.array(results['gradient_descent']['history']['lyapunov']),
             nest_objective=np.array(results['nesterov']['history']['objective']),
             nest_lyapunov=np.array(results['nesterov']['history']['lyapunov']),
             admm_objective=np.array(results['admm']['history']['objective']),
             admm_primal_residual=np.array(results['admm']['history']['primal_residual']),
             admm_dual_residual=np.array(results['admm']['history']['dual_residual']),
             admm_lyapunov=np.array(results['admm']['history']['lyapunov']),
             f_star=results['f_star'],
             L=problem.L,
             mu=problem.mu)
    print("    Saved: outputs/convergence_results.npz")
    
    # Generate all visualizations
    print("\n    Generating figures...")
    plot_convergence_comparison(results, 'report/images/convergence_comparison.png')
    plot_lyapunov_analysis(results, 'report/images/lyapunov_analysis.png')
    plot_phase_space(results, 'report/images/phase_space.png')
    plot_linear_convergence(results, 'report/images/linear_convergence.png')
    create_comparison_table(results, 'outputs/comparison_table.csv')
    
    print("\n" + "="*60)
    print("Experiments completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - report/images/data_overview.png")
    print("  - report/images/convergence_comparison.png")
    print("  - report/images/lyapunov_analysis.png")
    print("  - report/images/phase_space.png")
    print("  - report/images/linear_convergence.png")
    print("  - outputs/convergence_results.npz")
    print("  - outputs/comparison_table.csv")


if __name__ == '__main__':
    main()
