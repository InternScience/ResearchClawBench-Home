"""Parse the MATBG Superfluid Stiffness Core Dataset txt file into numpy arrays.

The dataset is provided as a human-readable text file with sections like:
  **<Field name>:**
  [ <numbers separated by whitespace, possibly multiline> ]

We parse it once and save a single .npz for downstream use.
"""
from __future__ import annotations

import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "MATBG Superfluid Stiffness Core Dataset.txt")
OUT = os.path.join(ROOT, "outputs", "matbg_data.npz")


def parse_dataset(path: str) -> dict:
    text = open(path, "r").read()

    # Find all "**Name:**" headers followed by content until the next header or EOF.
    # We are interested in array headers (those whose content starts with [).
    pattern = re.compile(r"\*\*([^*]+?)\*\*\s*", re.DOTALL)
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().rstrip(":")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end]

    arrays: dict[str, np.ndarray] = {}
    for name, body in sections.items():
        # Look for a bracketed numeric block.
        m = re.search(r"\[(.*?)\]", body, re.DOTALL)
        if not m:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
        if not nums:
            continue
        try:
            arr = np.array([float(x) for x in nums])
        except ValueError:
            continue
        arrays[name] = arr

    return arrays


def short_key(name: str) -> str:
    """Short safe key for npz."""
    name = name.lower()
    # extract content inside parentheses if any (e.g. "(D_s_conv)")
    m = re.search(r"\(([^)]+)\)", name)
    if m:
        return m.group(1).strip()
    return re.sub(r"[^a-z0-9_]+", "_", name).strip("_")


def main() -> None:
    arrays = parse_dataset(DATA)

    keyed: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        k = short_key(name)
        # avoid clobbering
        base = k
        i = 1
        while k in keyed:
            i += 1
            k = f"{base}_{i}"
        keyed[k] = arr

    print("Parsed arrays (key -> length):")
    for k, v in keyed.items():
        print(f"  {k:40s}  n={len(v):4d}  min={v.min():.4g}  max={v.max():.4g}")

    np.savez(OUT, **keyed)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
