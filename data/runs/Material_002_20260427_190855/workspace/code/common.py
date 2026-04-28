"""Common config for MACE-MP-0 reproduction experiments."""
import os

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_002_20260427_190855"
MODEL_PATH = os.path.join(WORKSPACE, "models", "MACE-MP-0b3-medium.model")
OUTPUTS = os.path.join(WORKSPACE, "outputs")
IMAGES = os.path.join(WORKSPACE, "report", "images")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(IMAGES, exist_ok=True)


def make_calc():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float32")
