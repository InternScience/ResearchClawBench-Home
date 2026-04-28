"""Stub module to allow unpickling of RealisticCrystalDataset objects.

The original RealisticCrystalDataset is a torch_geometric InMemoryDataset
subclass.  For our purposes we only need to access the underlying
``data_list`` of torch_geometric Data objects, so a minimal class with
``__init__`` accepting nothing and Python attribute storage is sufficient.
"""
from __future__ import annotations
from typing import Any, List


class RealisticCrystalDataset:
    """Minimal stand-in.  pickle's NEWOBJ + __setstate__ via __dict__ update
    will assign the persisted attributes (data_list, elem_to_idx, ...).
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - never used
        self.data_list: List[Any] = []

    # support the common InMemoryDataset-like interface
    def __len__(self) -> int:
        return len(getattr(self, "data_list", []))

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __iter__(self):
        return iter(self.data_list)
