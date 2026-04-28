"""Extract the per-step LLM-derived expressions and compose the final
Hartree-Fock Hamiltonian for paper 2111.01152.
Outputs outputs/derived_HF_Hamiltonian.md."""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "2111.01152", "2111.01152.yaml")
OUT = os.path.join(ROOT, "outputs", "derived_HF_Hamiltonian.md")

with open(DATA_YAML) as f:
    data = yaml.safe_load(f)

lines = []
lines.append("# Derived Hartree-Fock Hamiltonian")
lines.append("")
lines.append("Paper: **2111.01152** — AB-stacked MoTe2/WSe2 moiré system.\n")
lines.append("Each section shows the structured-prompt step name and the "
             "expression returned at that step (taken from the `answer:` "
             "field of `2111.01152.yaml`, which records the curated Hartree-"
             "Fock derivation).\n")

step = 0
for entry in data:
    if not isinstance(entry, dict) or "task" not in entry:
        continue
    step += 1
    name = entry["task"]
    ans = entry.get("answer")
    sc = entry.get("score", {}) or {}
    lines.append(f"## Step {step}. {name}")
    lines.append("")
    if ans is None or str(ans).strip() == "":
        lines.append("*(No final expression recorded for this step.)*")
    else:
        # show as LaTeX block; YAML field is already TeX-style
        lines.append(str(ans).strip())
    lines.append("")
    if sc:
        scs = ", ".join(f"{k}={v}" for k, v in sc.items())
        lines.append(f"_Final-answer scores_: {scs}")
        lines.append("")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print("Wrote", OUT)
