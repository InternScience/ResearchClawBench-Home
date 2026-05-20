import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

workspace = Path(__file__).resolve().parent.parent
npz_path = workspace / "outputs" / "parsed_data.npz"
img_dir = workspace / "report" / "images"
img_dir.mkdir(parents=True, exist_ok=True)

data = np.load(npz_path)

T_common = data["T"]
D_s_exp_temp = np.interp(T_common, np.linspace(0, 1.2, len(data["D_s_experimental"])), data["D_s_experimental"],
                         left=data["D_s_experimental"][0], right=data["D_s_experimental"][-1])
D_s_bcs = np.interp(T_common, np.linspace(0, 1.2, len(data["D_s_bcs"])), data["D_s_bcs"])
D_s_nodal = np.interp(T_common, np.linspace(0, 1.2, len(data["D_s_nodal"])), data["D_s_nodal"])
D_s_n2 = np.interp(T_common, np.linspace(0, 1.2, len(data["D_s_power_n2"])), data["D_s_power_n2"])
D_s_n25 = np.interp(T_common, np.linspace(0, 1.2, len(data["D_s_power_n2_5"])), data["D_s_power_n2_5"])
D_s_n3 = np.interp(T_common, np.linspace(0, 1.2, len(data["D_s_power_n3"])), data["D_s_power_n3"])

# Figure 7: Low-T zoom
mask = T_common <= 0.5
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(T_common[mask], D_s_bcs[mask], label="BCS (s-wave)", lw=2)
ax.plot(T_common[mask], D_s_nodal[mask], label="Nodal (linear)", lw=2)
ax.plot(T_common[mask], D_s_n2[mask], label="Power law n=2", lw=2, ls="--")
ax.plot(T_common[mask], D_s_n25[mask], label="Power law n=2.5", lw=2, ls="--")
ax.plot(T_common[mask], D_s_n3[mask], label="Power law n=3", lw=2, ls="--")
ax.plot(T_common[mask], D_s_exp_temp[mask], label="Experiment", lw=2, color="black", alpha=0.7)
ax.set_xlabel(r"Temperature $T$ (K)")
ax.set_ylabel(r"Superfluid stiffness $D_s$ (normalized)")
ax.set_title("Low-temperature superfluid stiffness ($T \\leq 0.5$ K)")
ax.legend(loc="upper right")
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig7_lowT_zoom.png", dpi=300)
plt.close(fig)

# Figure 8: Residual stiffness above T_c
mask = T_common >= 0.8
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(T_common[mask], D_s_bcs[mask], label="BCS (s-wave)", lw=2)
ax.plot(T_common[mask], D_s_nodal[mask], label="Nodal (linear)", lw=2)
ax.plot(T_common[mask], D_s_n2[mask], label="Power law n=2", lw=2, ls="--")
ax.plot(T_common[mask], D_s_n25[mask], label="Power law n=2.5", lw=2, ls="--")
ax.plot(T_common[mask], D_s_n3[mask], label="Power law n=3", lw=2, ls="--")
ax.plot(T_common[mask], D_s_exp_temp[mask], label="Experiment", lw=2, color="black", alpha=0.7)
ax.axvline(1.0, color="gray", ls="-.", lw=1, label=r"$T_c = 1.0$ K")
ax.set_xlabel(r"Temperature $T$ (K)")
ax.set_ylabel(r"Superfluid stiffness $D_s$ (normalized)")
ax.set_title("Superfluid stiffness above $T_c$")
ax.legend(loc="upper right")
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig8_above_Tc.png", dpi=300)
plt.close(fig)

print("Extra figures saved.")
