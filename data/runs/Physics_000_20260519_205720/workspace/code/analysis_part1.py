import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

os.makedirs("report/images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with open("outputs/parsed_data.json", "r") as f:
    d = json.load(f)

# Figure 1: Hexagonal lattice shell path
hex_coords = np.array(d["hexagonal_coords"])
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(hex_coords[:,0], hex_coords[:,1], c='steelblue', s=100, zorder=3)
for i, (x,y) in enumerate(hex_coords):
    ax.text(x+0.08, y+0.08, f"({x},{y})", fontsize=7)
# Draw a path example: (0,0)->(0,1)->(1,1)->(1,2)->(2,2)->(2,3)
path = [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3)]
path_arr = np.array(path)
ax.plot(path_arr[:,0], path_arr[:,1], 'r-o', markersize=4, linewidth=1.5, label='Example shell path')
ax.set_xlabel('Hexagonal coordinate h')
ax.set_ylabel('Hexagonal coordinate k')
ax.set_title('Hexagonal Lattice Coordinates and Shell Path')
ax.legend()
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.savefig("report/images/fig1_hexagonal_path.png", dpi=300)
plt.close(fig)

# Figure 2: Mackay vs new magic numbers
mackay = np.array(d["mackay_sequence"])
new_b5 = np.array(d["new_sequence_b5"])
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(np.arange(1, len(mackay)+1), mackay, 'o-', label='Mackay sequence', color='tab:blue')
ax.plot(np.arange(1, len(new_b5)+1), new_b5, 's-', label='New sequence (b=5)', color='tab:orange')
ax.set_xlabel('Shell index $i$')
ax.set_ylabel('Magic number $N_i$')
ax.set_title('Magic Number Sequences for Icosahedral Shells')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig("report/images/fig2_magic_numbers.png", dpi=300)
plt.close(fig)

# Save comparison table
magic_df = pd.DataFrame({"shell_index": np.arange(1, len(new_b5)+1)})
magic_df["mackay"] = list(mackay) + [np.nan]*(len(new_b5)-len(mackay))
magic_df["new_b5"] = list(new_b5)
magic_df.to_csv("outputs/magic_numbers_comparison.csv", index=False)

print("Part1 done: fig1, fig2 saved.")
