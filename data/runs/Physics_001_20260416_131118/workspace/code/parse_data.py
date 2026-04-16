import re
import json

with open("data/MATBG Superfluid Stiffness Core Dataset.txt", "r") as f:
    text = f.read()

# Let's find all array-like structures
# We can just split the file by sections
sections = re.split(r'\n\*\*(.*?)\*\*\n', text)

data = {}
current_key = None

for i in range(1, len(sections), 2):
    key = sections[i].strip()
    val = sections[i+1].strip()
    data[key] = val

# Save to a json for easier inspection
with open("outputs/parsed_data_raw.json", "w") as f:
    json.dump(data, f, indent=4)

