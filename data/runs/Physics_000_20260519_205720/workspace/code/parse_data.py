import re, json, ast, os

infile = "data/Multi-component Icosahedral Reproduction Data.txt"
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)

with open(infile, "r") as f:
    text = f.read()

# Remove comment lines starting with #
lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
clean_text = "\n".join(lines)

# Extract variable assignments using regex
pattern = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$", re.MULTILINE)
matches = pattern.findall(clean_text)

data = {}
for name, expr in matches:
    try:
        value = ast.literal_eval(expr)
        data[name] = value
    except Exception:
        try:
            # Allow safe eval for list multiplication like ['Na']*50
            value = eval(expr, {"__builtins__": {}}, {})
            data[name] = value
        except Exception as e2:
            print(f"Warning: could not parse {name}: {e2}")

# Save as JSON
with open(os.path.join(outdir, "parsed_data.json"), "w") as f:
    json.dump(data, f, indent=2)

# Also create a simple summary CSV of key scalar arrays
import csv

summary_rows = []
for key, val in data.items():
    summary_rows.append({"variable": key, "type": type(val).__name__, "length": len(val) if hasattr(val, "__len__") else "N/A", "sample": str(val)[:200]})

with open(os.path.join(outdir, "data_summary.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variable", "type", "length", "sample"])
    writer.writeheader()
    writer.writerows(summary_rows)

print("Parsed", len(data), "variables.")
for k in data:
    print(k, type(data[k]).__name__, len(data[k]) if hasattr(data[k], '__len__') else '')
