import ast
import re

def parse_data_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract variable assignments
    pattern = r'^([a-zA-Z0-9_]+)\s*=\s*(.+)$'
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            var_name = match.group(1)
            var_value_str = match.group(2)
            try:
                var_value = ast.literal_eval(var_value_str)
                data[var_name] = var_value
            except Exception as e:
                print(f"Error parsing {var_name}: {e}")
    return data

if __name__ == "__main__":
    data = parse_data_file('data/Multi-component Icosahedral Reproduction Data.txt')
    for key, value in data.items():
        print(f"{key}: {type(value)}")
