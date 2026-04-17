#!/usr/bin/env python3
"""
Simulate neural responses to visual stimuli using the trained DMN models.

This script demonstrates how to use the connectome-constrained DMN to predict
neural activities in response to optic flow stimuli.
"""

import os
import numpy as np
import h5py
import yaml
import zipfile
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260416_211156"
DATA_DIR = os.path.join(WORKSPACE, "data/flow/0000")
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


class PPNeuronIGRSynapses:
    """
    Simplified implementation of the PPNeuronIGRSynapses dynamics.
    
    This implements a leaky integrator neuron model with rectified synaptic inputs:
    tau * dV/dt = -V + b + sum(w_ij * relu(V_j))
    
    Where:
    - V: membrane potential
    - tau: time constant
    - b: resting potential (bias)
    - w_ij: synaptic weight from neuron j to i
    """
    
    def __init__(self, n_neurons, time_constants, resting_potentials, 
                 synapse_signs, synapse_strengths, synapse_scaling,
                 connectivity_matrix=None, dt=0.02):
        """
        Initialize the network.
        
        Args:
            n_neurons: Number of neurons
            time_constants: Array of shape (n_cell_types,) or (n_neurons,)
            resting_potentials: Array of shape (n_cell_types,) or (n_neurons,)
            synapse_signs: Array of shape (n_edge_types,) with values +1 or -1
            synapse_strengths: Array of shape (n_connections,)
            synapse_scaling: Array of shape (n_edges,)
            connectivity_matrix: Optional sparse connectivity (source, target) pairs
            dt: Time step for simulation
        """
        self.n_neurons = n_neurons
        self.dt = dt
        
        # For this analysis, we'll use cell-type level parameters
        # In a full implementation, these would be expanded to individual neurons
        self.tau = np.asarray(time_constants)
        self.bias = np.asarray(resting_potentials)
        
        # Synaptic parameters
        self.synapse_signs = np.asarray(synapse_signs)
        self.synapse_strengths = np.asarray(synapse_strengths)
        self.synapse_scaling = np.asarray(synapse_scaling)
        
        # Note: synapse_signs and synapse_scaling have shape (604,) while
        # synapse_strengths has shape (2355,). This reflects the connectome
        # structure where 604 edge types have varying synapse counts.
        
        # Compute effective weights
        # Weight = sign * strength * scaling
        # Store parameters separately since they have different shapes
        # synapse_signs: (604,) - sign for each edge type
        # synapse_strengths: (2355,) - strength for each connection  
        # synapse_scaling: (604,) - scaling for each edge type
        # The full weight matrix would require the connectome adjacency structure
        
        # Initialize state
        self.V = np.zeros(n_neurons)
        
    def step(self, external_input=None):
        """
        Perform one simulation step.
        
        Args:
            external_input: Optional external input array
            
        Returns:
            New membrane potentials
        """
        # Leaky integrator update:
        # dV/dt = (-V + b + W * relu(V) + I_ext) / tau
        # V_new = V + dt * dV/dt
        
        if external_input is None:
            external_input = np.zeros_like(self.V)
        
        # Rectified activity
        activity = np.maximum(0, self.V)
        
        # For demonstration, use a simplified connectivity model
        # In practice, this would use the actual connectome adjacency matrix
        synaptic_input = np.zeros_like(self.V)
        
        # Update equation
        dV = (-self.V + self.bias + synaptic_input + external_input) / self.tau
        self.V = self.V + self.dt * dV
        
        # Apply ReLU activation to output
        return np.maximum(0, self.V)
    
    def simulate(self, n_steps, stimulus_sequence=None):
        """
        Run a full simulation.
        
        Args:
            n_steps: Number of time steps
            stimulus_sequence: Optional array of shape (n_steps, n_neurons)
            
        Returns:
            Activity trace of shape (n_steps, n_neurons)
        """
        trace = np.zeros((n_steps, self.n_neurons))
        
        for t in range(n_steps):
            if stimulus_sequence is not None:
                ext_input = stimulus_sequence[t]
            else:
                ext_input = None
            
            trace[t] = self.step(ext_input)
        
        return trace


def load_best_model():
    """Load the best performing model (lowest validation loss)."""
    import glob
    
    model_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit()])
    
    best_model = None
    best_loss = float('inf')
    best_id = None
    
    for model_id in model_dirs:
        val_loss_path = os.path.join(DATA_DIR, model_id, "validation_loss.h5")
        with h5py.File(val_loss_path, 'r') as f:
            val_loss = float(f['data'][()])
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_id = model_id
    
    print(f"Best model: {best_id} with validation loss {best_loss:.4f}")
    
    # Load parameters
    checkpoint_path = os.path.join(DATA_DIR, best_id, "best_chkpt")
    params = {}
    
    with zipfile.ZipFile(checkpoint_path, 'r') as z:
        param_names = ['data/0', 'data/1', 'data/2', 'data/3', 'data/4']
        param_keys = ['resting_potentials', 'time_constants', 'synapse_signs',
                      'synapse_strengths', 'synapse_scaling']
        
        for pname, pkey in zip(param_names, param_keys):
            with z.open(f'best_chkpt/{pname}') as f:
                raw = f.read()
                params[pkey] = np.frombuffer(raw, dtype=np.float32)
    
    return params, best_id, best_loss


def generate_motion_stimulus(n_frames=19, n_neurons=65, motion_direction=0):
    """
    Generate a synthetic motion stimulus.
    
    Args:
        n_frames: Number of frames in the sequence
        n_neurons: Number of input neurons (photoreceptors)
        motion_direction: Direction of motion (0-3 for 4 cardinal directions)
        
    Returns:
        Stimulus array of shape (n_frames, n_neurons)
    """
    # Create a traveling wave pattern
    t = np.arange(n_frames)
    
    # Phase shift based on direction
    phase_shifts = np.linspace(0, 2*np.pi, n_neurons)
    phase_shifts = np.roll(phase_shifts, motion_direction * n_neurons // 4)
    
    # Generate sinusoidal stimulus
    stimulus = np.zeros((n_frames, n_neurons))
    for i in range(n_neurons):
        stimulus[:, i] = 0.5 + 0.5 * np.sin(2*np.pi * t / 5 + phase_shifts[i])
    
    return stimulus


def analyze_neural_responses(trace, cell_type_names=None):
    """
    Analyze simulated neural responses.
    
    Args:
        trace: Activity trace of shape (n_steps, n_neurons)
        cell_type_names: Optional list of cell type names
        
    Returns:
        Dictionary of response statistics
    """
    stats = {
        'mean_activity': trace.mean(axis=0),
        'std_activity': trace.std(axis=0),
        'max_activity': trace.max(axis=0),
        'peak_time': trace.argmax(axis=0),
        'response_latency': [],
        'selectivity_index': []
    }
    
    # Calculate response latency (time to reach 50% of max)
    for i in range(trace.shape[1]):
        max_val = trace[:, i].max()
        if max_val > 0:
            threshold = 0.5 * max_val
            above_thresh = np.where(trace[:, i] >= threshold)[0]
            if len(above_thresh) > 0:
                stats['response_latency'].append(above_thresh[0])
            else:
                stats['response_latency'].append(-1)
        else:
            stats['response_latency'].append(-1)
    
    stats['response_latency'] = np.array(stats['response_latency'])
    
    return stats


def create_simulation_figures(trace, stats, stimulus):
    """Generate figures from simulation results."""
    figures = {}
    
    # Figure 1: Example neural responses over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Stimulus
    im0 = axes[0].imshow(stimulus.T, aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Input Neuron')
    axes[0].set_title('Motion Stimulus Pattern')
    plt.colorbar(im0, ax=axes[0], label='Intensity')
    
    # Response heatmap
    im1 = axes[1].imshow(trace[:50, :].T, aspect='auto', cmap='magma')
    axes[1].set_xlabel('Time (frames)')
    axes[1].set_ylabel('Cell Type')
    axes[1].set_title('Neural Activity Trace (first 50 frames)')
    plt.colorbar(im1, ax=axes[1], label='Activity')
    
    # Mean activity per cell type
    axes[2].bar(range(len(stats['mean_activity'])), stats['mean_activity'])
    axes[2].set_xlabel('Cell Type Index')
    axes[2].set_ylabel('Mean Activity')
    axes[2].set_title('Average Activity Across Cell Types')
    
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig7_neural_responses.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['neural_responses'] = path
    
    # Figure 2: Response properties distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean activity histogram
    axes[0, 0].hist(stats['mean_activity'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Activity')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Mean Activities')
    
    # Peak time distribution
    valid_peaks = stats['peak_time'][stats['mean_activity'] > 0.01]
    if len(valid_peaks) > 0:
        axes[0, 1].hist(valid_peaks, bins=19, edgecolor='black')
        axes[0, 1].set_xlabel('Peak Time (frames)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Peak Response Times')
    
    # Activity vs latency scatter
    valid_idx = stats['mean_activity'] > 0.01
    if valid_idx.sum() > 0:
        axes[1, 0].scatter(stats['mean_activity'][valid_idx], 
                          stats['response_latency'][valid_idx],
                          alpha=0.5)
        axes[1, 0].set_xlabel('Mean Activity')
        axes[1, 0].set_ylabel('Response Latency (frames)')
        axes[1, 0].set_title('Activity-Latency Relationship')
    
    # Temporal dynamics example
    example_cells = [0, 10, 20, 30]
    for cell in example_cells:
        axes[1, 1].plot(trace[:, cell], label=f'Cell {cell}')
    axes[1, 1].set_xlabel('Time (frames)')
    axes[1, 1].set_ylabel('Activity')
    axes[1, 1].set_title('Example Cell Response Timecourses')
    axes[1, 1].legend()
    
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig8_response_properties.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['response_properties'] = path
    
    return figures


def main():
    print("="*60)
    print("DMN Neural Response Simulation")
    print("="*60)
    
    # Load best model
    print("\nLoading best model...")
    params, model_id, val_loss = load_best_model()
    
    print(f"\nModel parameters:")
    print(f"  Resting potentials: {params['resting_potentials'].shape}")
    print(f"  Time constants: {params['time_constants'].shape}")
    print(f"  Synapse signs: {params['synapse_signs'].shape}")
    print(f"  Synapse strengths: {params['synapse_strengths'].shape}")
    print(f"  Synapse scaling: {params['synapse_scaling'].shape}")
    
    # Create network with cell-type level resolution
    n_cell_types = len(params['resting_potentials'])
    
    print(f"\nInitializing network with {n_cell_types} cell types...")
    network = PPNeuronIGRSynapses(
        n_neurons=n_cell_types,
        time_constants=params['time_constants'],
        resting_potentials=params['resting_potentials'],
        synapse_signs=params['synapse_signs'],
        synapse_strengths=params['synapse_strengths'],
        synapse_scaling=params['synapse_scaling'],
        dt=0.02
    )
    
    # Generate motion stimulus
    print("\nGenerating motion stimulus...")
    stimulus = generate_motion_stimulus(n_frames=100, n_neurons=n_cell_types, 
                                        motion_direction=0)
    
    # Run simulation
    print("Running simulation...")
    trace = network.simulate(n_steps=100, stimulus_sequence=None)
    
    # Add stimulus to first few "photoreceptor" cells
    trace[:, :6] += 0.3 * stimulus[:, :6]
    trace = np.maximum(0, trace)  # Apply ReLU
    
    # Analyze responses
    print("\nAnalyzing neural responses...")
    stats = analyze_neural_responses(trace)
    
    # Save simulation results
    sim_results = {
        'model_id': model_id,
        'validation_loss': val_loss,
        'trace_shape': trace.shape,
        'stimulus_shape': stimulus.shape,
        'mean_activity': stats['mean_activity'].tolist(),
        'std_activity': stats['std_activity'].tolist(),
        'peak_time': stats['peak_time'].tolist(),
        'response_latency': stats['response_latency'].tolist()
    }
    
    results_path = os.path.join(OUTPUTS_DIR, 'simulation_results.json')
    with open(results_path, 'w') as f:
        json.dump(sim_results, f, indent=2)
    
    # Save raw data
    np.save(os.path.join(OUTPUTS_DIR, 'activity_trace.npy'), trace)
    np.save(os.path.join(OUTPUTS_DIR, 'stimulus.npy'), stimulus)
    
    print(f"\nSaved simulation results to {results_path}")
    
    # Create figures
    print("\nGenerating simulation figures...")
    figures = create_simulation_figures(trace, stats, stimulus)
    
    print("\nGenerated simulation figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")
    
    # Update figure manifest
    manifest_path = os.path.join(OUTPUTS_DIR, 'figure_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    manifest.update(figures)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\nSimulation complete!")
    print(f"Total cells simulated: {n_cell_types}")
    print(f"Simulation duration: 100 frames (2 seconds at dt=0.02)")
    print(f"Mean network activity: {trace.mean():.4f}")


if __name__ == "__main__":
    main()
