"""H0 Distance Network Analysis"""
import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
import re

c_km = 299792.458

@dataclass
class Anchor:
    name: str
    mu: float
    err: float

@dataclass
class HostMeasurement:
    host: str
    method: str
    anchor: str
    mu_meas: float
    err_meas: float

@dataclass
class SNECalibrator:
    host: str
    mB: float
    err_mB: float

@dataclass
class SBFCalibrator:
    host: str
    mF110W: float
    err_mF110W: float

@dataclass
class HubbleFlowSN:
    z: float
    mB: float
    err_mB: float
    v_pec_err: float

@dataclass
class HubbleFlowSBF:
    z: float
    mF110W: float
    err_mF110W: float
    v_pec_err: float

def parse_dataset(filepath):
    data = {
        "anchors": {},
        "host_measurements": [],
        "sneia_calibrators": [],
        "sbf_calibrators": [],
        "hubble_flow_sneia": [],
        "hubble_flow_sbf": [],
        "method_anchor_err": {},
        "host_group": {},
        "depth_scatter": 0.10
    }
    with open(filepath, "r") as f:
        content = f.read()
    
    anchor_match = re.search(r"anchors = \{([^}]+)\}", content, re.DOTALL)
    if anchor_match:
        for match in re.finditer(r"'([^']+)': \{'mu': ([\d.]+), 'err': ([\d.]+)\}", anchor_match.group(1)):
            data["anchors"][match.group(1)] = Anchor(match.group(1), float(match.group(2)), float(match.group(3)))
    
    host_pattern = r"\('([^']+)', '([^']+)', '([^']+)', ([\d.]+), ([\d.]+)\)"
    host_section = re.search(r"host_measurements = \[([^\]]+)\]", content, re.DOTALL)
    if host_section:
        for match in re.finditer(host_pattern, host_section.group(1)):
            data["host_measurements"].append(HostMeasurement(match.group(1), match.group(2), match.group(3), float(match.group(4)), float(match.group(5))))
    
    sne_section = re.search(r"sneia_calibrators = \[([^\]]+)\]", content, re.DOTALL)
    if sne_section:
        for match in re.finditer(host_pattern, sne_section.group(1)):
            data["sneia_calibrators"].append(SNECalibrator(match.group(1), float(match.group(2)), float(match.group(3))))
    
    sbf_section = re.search(r"sbf_calibrators = \[([^\]]+)\]", content, re.DOTALL)
    if sbf_section:
        for match in re.finditer(host_pattern, sbf_section.group(1)):
            data["sbf_calibrators"].append(SBFCalibrator(match.group(1), float(match.group(2)), float(match.group(3))))
    
    hflow_pattern = r"\(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)"
    hflow_sne = re.search(r"hubble_flow_sneia = \[([^\]]+)\]", content, re.DOTALL)
    if hflow_sne:
        for match in re.finditer(hflow_pattern, hflow_sne.group(1)):
            data["hubble_flow_sneia"].append(HubbleFlowSN(float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))))
    
    hflow_sbf = re.search(r"hubble_flow_sbf = \[([^\]]+)\]", content, re.DOTALL)
    if hflow_sbf:
        for match in re.finditer(hflow_pattern, hflow_sbf.group(1)):
            data["hubble_flow_sbf"].append(HubbleFlowSBF(float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))))
    
    method_err = re.search(r"method_anchor_err = \{([^}]+)\}", content, re.DOTALL)
    if method_err:
        for match in re.finditer(r"\('([^']+)', '([^']+)'\): ([\d.]+)", method_err.group(1)):
            data["method_anchor_err"][(match.group(1), match.group(2))] = float(match.group(3))
    
    host_group = re.search(r"host_group = \{([^}]+)\}", content, re.DOTALL)
    if host_group:
        for match in re.finditer(r"'([^']+)': '([^']+)'", host_group.group(1)):
            data["host_group"][match.group(1)] = match.group(2)
    
    return data

class DistanceNetwork:
    def __init__(self, data):
        self.data = data
        self.H0 = None
        self.H0_err = None
        self.chi2 = None
        self.dof = None
        self.individual_H0 = []
        self.individual_H0_err = []
        self.individual_labels = []
        self.MB_mean = None
        self.MB_mean_err = None

    def fit_H0(self):
        H0_values = []
        H0_weights = []
        self.individual_H0 = []
        self.individual_H0_err = []
        self.individual_labels = []
        
        M_B_values = []
        M_B_errs = []
        
        for sne in self.data["sneia_calibrators"]:
            host_dists = [h for h in self.data["host_measurements"] if h.host == sne.host]
            if host_dists:
                mu_hosts = []
                mu_errs = []
                for hd in host_dists:
                    mu_hosts.append(hd.mu_meas)
                    anchor_err = self.data["anchors"][hd.anchor].err
                    method_err = self.data["method_anchor_err"].get((hd.method, hd.anchor), 0.0)
                    total_err = np.sqrt(hd.err_meas**2 + anchor_err**2 + method_err**2)
                    mu_errs.append(total_err)
                weights = [1/e**2 for e in mu_errs]
                mu_host = np.average(mu_hosts, weights=weights)
                mu_host_err = np.sqrt(1/sum(weights))
                M_B = sne.mB - mu_host
                M_B_err = np.sqrt(sne.err_mB**2 + mu_host_err**2)
                M_B_values.append(M_B)
                M_B_errs.append(M_B_err)
        
        weights = [1/e**2 for e in M_B_errs]
        self.MB_mean = np.average(M_B_values, weights=weights)
        self.MB_mean_err = np.sqrt(1/sum(weights))
        
        for i, hf in enumerate(self.data["hubble_flow_sneia"]):
            mu = hf.mB - self.MB_mean
            mu_err = np.sqrt(hf.err_mB**2 + self.MB_mean_err**2)
            v_pec_mag_err = 5/np.log(10) * hf.v_pec_err/(c_km * hf.z)
            total_mu_err = np.sqrt(mu_err**2 + v_pec_mag_err**2)
            d_L = 10**((mu - 25)/5)
            H0_i = c_km * hf.z / d_L
            dH0_dmu = -H0_i * np.log(10)/5
            H0_i_err = abs(dH0_dmu) * total_mu_err
            H0_values.append(H0_i)
            H0_weights.append(1/H0_i_err**2)
            self.individual_H0.append(H0_i)
            self.individual_H0_err.append(H0_i_err)
            self.individual_labels.append(f"SNeIa_z{hf.z:.3f}")
        
        H0_array = np.array(H0_values)
        weights_array = np.array(H0_weights)
        self.H0 = np.average(H0_array, weights=weights_array)
        self.H0_err = np.sqrt(1/np.sum(weights_array))
        residuals = H0_array - self.H0
        self.chi2 = np.sum(weights_array * residuals**2)
        self.dof = len(H0_values) - 1
        
        return self.H0, self.H0_err, self.chi2, self.dof

def create_visualizations(data, network, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    
    # Data Overview Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Anchors
    ax1 = axes[0, 0]
    anchor_names = list(data["anchors"].keys())
    anchor_mus = [data["anchors"][a].mu for a in anchor_names]
    anchor_errs = [data["anchors"][a].err for a in anchor_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(anchor_names, anchor_mus, yerr=anchor_errs, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Distance Modulus mu (mag)')
    ax1.set_title('Geometric Distance Anchors', fontweight='bold')
    ax1.set_ylim(0, 32)
    for bar, mu, err in zip(bars, anchor_mus, anchor_errs):
        ax1.text(bar.get_x() + bar.get_width()/2, mu + err + 0.5, f'{mu:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Host measurements
    ax2 = axes[0, 1]
    host_data = {}
    for h in data["host_measurements"]:
        if h.host not in host_data:
            host_data[h.host] = {'methods': [], 'mu_vals': [], 'mu_errs': []}
        host_data[h.host]['methods'].append(f"{h.method}({h.anchor})")
        host_data[h.host]['mu_vals'].append(h.mu_meas)
        host_data[h.host]['mu_errs'].append(h.err_meas)
    
    x_pos = []
    labels = []
    vals = []
    errs = []
    method_colors = {'Cepheid': '#1f77b4', 'TRGB': '#ff7f0e'}
    colors_list = []
    pos = 0
    for host, info in host_data.items():
        for method, mu, err in zip(info['methods'], info['mu_vals'], info['mu_errs']):
            x_pos.append(pos)
            labels.append(f"{host}\n{method}")
            vals.append(mu)
            errs.append(err)
            method_key = method.split('(')[0]
            colors_list.append(method_colors.get(method_key, '#999999'))
            pos += 1
        pos += 0.5
    
    ax2.errorbar(x_pos, vals, yerr=errs, fmt='o', capsize=3, color='black', markersize=6)
    ax2.scatter(x_pos, vals, c=colors_list, s=100, zorder=5, edgecolors='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Distance Modulus mu (mag)')
    ax2.set_title('Host Galaxy Distance Measurements', fontweight='bold')
    ax2.set_ylim(28, 33.5)
    
    # SNe Ia Calibrators
    ax3 = axes[1, 0]
    sne_hosts = [s.host for s in data["sneia_calibrators"]]
    sne_mB = [s.mB for s in data["sneia_calibrators"]]
    sne_err = [s.err_mB for s in data["sneia_calibrators"]]
    ax3.errorbar(range(len(sne_hosts)), sne_mB, yerr=sne_err, fmt='s', capsize=4, color='#d62728', markersize=8)
    ax3.set_xticks(range(len(sne_hosts)))
    ax3.set_xticklabels(sne_hosts, rotation=45, ha='right')
    ax3.set_ylabel('Apparent Magnitude m_B')
    ax3.set_title('Type Ia Supernova Calibrators', fontweight='bold')
    ax3.set_ylim(9, 13)
    
    # Hubble Flow
    ax4 = axes[1, 1]
    z_sne = [h.z for h in data["hubble_flow_sneia"]]
    mB_sne = [h.mB for h in data["hubble_flow_sneia"]]
    z_sbf = [h.z for h in data["hubble_flow_sbf"]]
    mF_sbf = [h.mF110W for h in data["hubble_flow_sbf"]]
    ax4.scatter(z_sne, mB_sne, c='#d62728', s=100, marker='s', label='SNe Ia', edgecolors='black', zorder=5)
    ax4.scatter(z_sbf, mF_sbf, c='#9467bd', s=100, marker='^', label='SBF', edgecolors='black', zorder=5)
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Apparent Magnitude')
    ax4.set_title('Hubble Flow Measurements', fontweight='bold')
    ax4.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/data_overview.png")

def create_hubble_diagram(data, network, output_dir):
    # Hubble Diagram
    fig, ax = plt.subplots(figsize=(10, 7))
    H0 = network.H0 if network.H0 else 73.0
    
    z_plot = np.linspace(0.02, 0.09, 100)
    mu_theory = 5 * np.log10(c_km * z_plot / H0 * (1 + 0.225 * z_plot)) + 25
    
    for i, hf in enumerate(data["hubble_flow_sneia"]):
        mu_obs = hf.mB - network.MB_mean
        mu_err = np.sqrt(hf.err_mB**2 + network.MB_mean_err**2)
        v_pec_mag_err = 5/np.log(10) * hf.v_pec_err/(c_km * hf.z)
        total_err = np.sqrt(mu_err**2 + v_pec_mag_err**2)
        ax.errorbar(hf.z, mu_obs, yerr=total_err, fmt='o', color='#1f77b4', markersize=10, capsize=4, zorder=5)
    
    ax.plot(z_plot, mu_theory, 'r--', linewidth=2, label=f'H0 = {H0:.2f} km/s/Mpc')
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Distance Modulus mu (mag)', fontsize=12)
    ax.set_title('Hubble Diagram - SNe Ia', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hubble_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/hubble_diagram.png")

def create_h0_results(network, output_dir):
    # H0 Results Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Individual H0 measurements
    ax1 = axes[0]
    H0_vals = network.individual_H0
    H0_errs = network.individual_H0_err
    labels = [f"z={l.split('z')[1]}" for l in network.individual_labels]
    
    y_pos = np.arange(len(H0_vals))
    ax1.errorbar(H0_vals, y_pos, xerr=H0_errs, fmt='o', capsize=4, color='#2ca02c', markersize=8)
    ax1.axvline(network.H0, color='red', linestyle='--', linewidth=2, label=f'H0 = {network.H0:.2f} ± {network.H0_err:.2f}')
    ax1.axvspan(network.H0 - network.H0_err, network.H0 + network.H0_err, alpha=0.2, color='red')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('H0 (km/s/Mpc)', fontsize=11)
    ax1.set_title('Individual H0 Measurements', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(65, 85)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Comparison with Planck
    ax2 = axes[1]
    measurements = ['This Work\n(SNe Ia)', 'Planck 2018\n(CMB)', 'SH0ES 2022\n(Cepheids+SNe)']
    H0_values = [network.H0, 67.4, 73.04]
    H0_errors = [network.H0_err, 0.5, 1.04]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    
    for i, (meas, val, err, col) in enumerate(zip(measurements, H0_values, H0_errors, colors)):
        ax2.errorbar([i], [val], yerr=[err], fmt='s', markersize=15, capsize=6, color=col, label=meas, zorder=5)
    
    ax2.axhspan(67.4 - 0.5, 67.4 + 0.5, alpha=0.15, color='#ff7f0e', label='Planck ±1σ')
    ax2.set_xticks(range(len(measurements)))
    ax2.set_xticklabels(measurements)
    ax2.set_ylabel('H0 (km/s/Mpc)', fontsize=11)
    ax2.set_title('Hubble Constant Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(65, 78)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/h0_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/h0_results.png")

def create_distance_ladder(data, network, output_dir):
    # Distance Ladder Schematic
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Local Distance Network - H0 Measurement', fontsize=16, ha='center', fontweight='bold')
    
    # Anchors
    anchor_box = dict(boxstyle='round,pad=0.5', facecolor='#e6f3ff', edgecolor='#1f77b4', linewidth=2)
    ax.text(1.5, 7.5, 'Geometric Anchors\n(Rung 0)', fontsize=12, ha='center', va='center', bbox=anchor_box, fontweight='bold')
    anchor_text = 'NGC 4258 (masers)\nLMC (DEBs)\nMilky Way (parallaxes)'
    ax.text(1.5, 6, anchor_text, fontsize=9, ha='center', va='top')
    
    # Primary indicators
    primary_box = dict(boxstyle='round,pad=0.5', facecolor='#fff2e6', edgecolor='#ff7f0e', linewidth=2)
    ax.text(5, 7.5, 'Primary Indicators\n(Rung 1)', fontsize=12, ha='center', va='center', bbox=primary_box, fontweight='bold')
    primary_text = 'Cepheids (P-L relation)\nTRGB (Tip magnitude)\nMiras, JAGB'
    ax.text(5, 6, primary_text, fontsize=9, ha='center', va='top')
    
    # Secondary indicators
    secondary_box = dict(boxstyle='round,pad=0.5', facecolor='#e6ffe6', edgecolor='#2ca02c', linewidth=2)
    ax.text(8.5, 7.5, 'Secondary Indicators\n(Rung 2)', fontsize=12, ha='center', va='center', bbox=secondary_box, fontweight='bold')
    secondary_text = 'SNe Ia (standardized)\nSBF, SNe II, FP, TF'
    ax.text(8.5, 6, secondary_text, fontsize=9, ha='center', va='top')
    
    # Hubble Flow
    hubble_box = dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', edgecolor='#d62728', linewidth=2)
    ax.text(5, 3, 'Hubble Flow\n(Rung 3)', fontsize=12, ha='center', va='center', bbox=hubble_box, fontweight='bold')
    hubble_text = f'z = 0.02 - 0.10\nH0 = {network.H0:.2f} ± {network.H0_err:.2f} km/s/Mpc'
    ax.text(5, 1.5, hubble_text, fontsize=10, ha='center', va='top', fontweight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(3.5, 7.5), xytext=(2.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 7.5), xytext=(5.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4), xytext=(5, 5.5), arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distance_ladder.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/distance_ladder.png")

def main():
    import os
    
    data_dir = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_002_20260415_115310/data"
    output_dir = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_002_20260415_115310/outputs"
    report_dir = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_002_20260415_115310/report/images"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print("="*60)
    print("H0 Distance Network Analysis")
    print("="*60)
    
    print("\n[1] Parsing dataset...")
    data = parse_dataset(f"{data_dir}/H0DN_MinimalDataset.txt")
    
    print(f"   - Anchors: {len(data['anchors'])}")
    print(f"   - Host measurements: {len(data['host_measurements'])}")
    print(f"   - SNe Ia calibrators: {len(data['sneia_calibrators'])}")
    print(f"   - Hubble flow SNe Ia: {len(data['hubble_flow_sneia'])}")
    print(f"   - Hubble flow SBF: {len(data['hubble_flow_sbf'])}")
    
    print("\n[2] Running GLS fit...")
    network = DistanceNetwork(data)
    H0, H0_err, chi2, dof = network.fit_H0()
    
    print(f"\n   RESULTS:")
    print(f"   - H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"   - chi2/dof = {chi2:.2f}/{dof}")
    print(f"   - M_B = {network.MB_mean:.2f} ± {network.MB_mean_err:.2f}")
    
    print("\n[3] Creating visualizations...")
    create_visualizations(data, network, report_dir)
    create_hubble_diagram(data, network, report_dir)
    create_h0_results(network, report_dir)
    create_distance_ladder(data, network, report_dir)
    
    print("\n[4] Saving results...")
    results = {
        "H0": float(H0),
        "H0_err": float(H0_err),
        "chi2": float(chi2),
        "dof": int(dof),
        "MB": float(network.MB_mean),
        "MB_err": float(network.MB_mean_err),
        "individual_H0": [(l, float(v), float(e)) for l, v, e in zip(network.individual_labels, network.individual_H0, network.individual_H0_err)]
    }
    
    with open(f"{output_dir}/h0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"   - Saved: {output_dir}/h0_results.json")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()
