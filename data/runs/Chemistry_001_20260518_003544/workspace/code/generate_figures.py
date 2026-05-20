import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_001_20260518_003544/report/images', exist_ok=True)

# Load predicted pose
pose = np.load('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_001_20260518_003544/outputs/predicted_ligand_pose.npy')

# Simple 3D scatter of predicted pose
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='blue', s=50, label='Predicted Pose')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Predicted Ligand Pose (3D)')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_001_20260518_003544/report/images/figure1_predicted_pose_3d.png', dpi=150)
plt.close()

# 2D projections
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(pose[:, 0], pose[:, 1], c='blue', s=50)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('XY Projection')
axes[1].scatter(pose[:, 0], pose[:, 2], c='blue', s=50)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title('XZ Projection')
axes[2].scatter(pose[:, 1], pose[:, 2], c='blue', s=50)
axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_title('YZ Projection')
plt.suptitle('2D Projections of Predicted Ligand Pose')
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_001_20260518_003544/report/images/figure2_projections.png', dpi=150)
plt.close()

print("Figures generated successfully.")