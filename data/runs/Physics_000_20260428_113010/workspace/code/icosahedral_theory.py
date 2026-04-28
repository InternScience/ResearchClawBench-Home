"""
Core theory for multi-component icosahedral shells.

A shell is indexed by Goldberg / Caspar–Klug-like indices (h,k) on a triangular
(hexagonal) lattice.  The triangulation number is

        T(h,k) = h^2 + h*k + k^2

For an icosahedral cap with 20 triangular faces, the number of vertices on a
shell of index (h,k) is

        N(h,k) = 10 * T(h,k) + 2          (full closed icosahedral shell)

The Mackay magic numbers correspond to the achiral (h, 0) family:
   T(1,0)=1   -> 12  (+1 centre = 13)
   T(2,0)=4   -> 42  (+13 inner = 55)
   T(3,0)=9   -> 92  (+55 inner = 147)
   T(4,0)=16  -> 162 (+147 inner = 309)

Chiral shells correspond to (h,k) with h != k and k>0.
Bergman / "BG" type corresponds to (h,h).

This module implements the geometry and the size-mismatch theory between
adjacent shells.  The optimal size ratio for two icosahedral shells of indices
T_i and T_{i+1} is

        rho_opt = sqrt( T_{i+1} / T_i ) * geometric_factor

where the geometric factor accounts for icosahedral curvature and equals
1 for ideal achiral->achiral stacking.  The relative mismatch between two
shells with atomic radii r_a and r_b is

        sm = | rho_opt - r_b/r_a | / rho_opt
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List

PHI = (1 + math.sqrt(5)) / 2  # golden ratio (icosahedral symmetry)


def triangulation(h: int, k: int) -> int:
    return h * h + h * k + k * k


def shell_count(h: int, k: int) -> int:
    """Number of atoms on the (h,k) icosahedral shell (excluding interior)."""
    if h == 0 and k == 0:
        return 1
    return 10 * triangulation(h, k) + 2


def cumulative_count(path: List[Tuple[int, int]]) -> int:
    """Total atoms in cluster after stacking `path` shells (inner-out)."""
    n = 0
    for (h, k) in path:
        n += shell_count(h, k)
    return n


def chiral_label(h: int, k: int) -> str:
    """Classify a shell using the dataset's labelling convention."""
    if h == 0 and k == 0:
        return "MC"  # Mackay centre (single atom)
    if k == 0:
        return "MC"  # Mackay (achiral, on-edge)
    if h == k:
        return "BG"  # Bergman (icosidodecahedral, achiral)
    # Pure chiral shells (h,k) with h!=k, k>0; categorise by k
    return f"Ch{k}"


def shell_radius(h: int, k: int, a: float = 1.0) -> float:
    """Effective radius of an icosahedral (h,k) shell in units of nearest-neighbour distance a.

    Uses the Goldberg-polyhedron circumradius formula approximated by
        R(h,k) = a * sqrt(T) / (2 sin(pi/5))
    which reduces to the canonical icosahedron radius for T=1.
    """
    return a * math.sqrt(triangulation(h, k)) / (2.0 * math.sin(math.pi / 5.0))


def optimal_size_ratio(T_i: int, T_ip1: int) -> float:
    """Theoretical optimal radius ratio between an inner (T_i) and outer (T_{i+1}) shell."""
    return math.sqrt(T_ip1 / T_i)


def size_mismatch(r_inner: float, r_outer: float, T_i: int, T_ip1: int) -> float:
    """Relative size mismatch sm between a pair of atom species occupying
    adjacent icosahedral shells.

    sm = |rho_opt - r_outer/r_inner| / rho_opt
    """
    rho_opt = optimal_size_ratio(T_i, T_ip1)
    rho = r_outer / r_inner
    return abs(rho_opt - rho) / rho_opt


def lj_energy(r: float, eps: float, sigma: float) -> float:
    """Pairwise Lennard-Jones potential."""
    sr6 = (sigma / r) ** 6
    return 4.0 * eps * (sr6 * sr6 - sr6)


@dataclass
class Shell:
    h: int
    k: int

    @property
    def T(self) -> int:
        return triangulation(self.h, self.k)

    @property
    def n_atoms(self) -> int:
        return shell_count(self.h, self.k)

    @property
    def label(self) -> str:
        return chiral_label(self.h, self.k)


def build_path_atoms(path: List[Tuple[int, int]]) -> List[int]:
    """Cumulative atom counts shell by shell."""
    out, total = [], 0
    for (h, k) in path:
        total += shell_count(h, k)
        out.append(total)
    return out


# ----- helper used by analysis scripts -----
SHELL_FAMILIES = [
    ("MC", [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]),
    ("BG", [(1, 1), (2, 2), (3, 3)]),
    ("Ch1", [(1, 2), (2, 3), (3, 4)]),
    ("Ch2", [(1, 3), (2, 4), (3, 5)]),
    ("Ch3", [(1, 4), (2, 5), (3, 6)]),
]


if __name__ == "__main__":
    print("Mackay (h,0) magic numbers:")
    cum = 1
    print(f"  centre: cum={cum}")
    for h in range(1, 6):
        n = shell_count(h, 0)
        cum += n
        print(f"  ({h},0) T={triangulation(h,0):2d}  shell={n:4d}  cum={cum}")
    print()
    print("Sample radii (a=1):")
    for (h, k) in [(1, 0), (2, 0), (1, 1), (1, 2), (2, 3)]:
        s = Shell(h, k)
        print(f"  ({h},{k})  label={s.label:4s}  T={s.T:2d}  N={s.n_atoms:4d}  R={shell_radius(h,k):.4f}")
