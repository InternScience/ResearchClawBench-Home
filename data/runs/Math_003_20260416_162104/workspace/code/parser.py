import sys

def parse_problem(line):
    if '?' not in line:
        return None
    premises_str, goal_str = line.split('?')
    premises = [p.strip() for p in premises_str.split(';') if p.strip()]
    goal = goal_str.strip()
    return premises, goal

def parse_rules(filepath):
    rules = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=>' not in line:
                continue
            ant_str, cons_str = line.split('=>')
            antecedents = [a.strip() for a in ant_str.split(',') if a.strip()]
            consequents = [c.strip() for c in cons_str.split(',') if c.strip()]
            rules.append((antecedents, consequents))
    return rules

if __name__ == '__main__':
    with open('data/imo_ag_30.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        parsed = parse_problem(line)
        if parsed:
            p, g = parsed
            print("Premises:", p)
            print("Goal:", g)
            break
    
    rules = parse_rules('data/rules.txt')
    print("Rule 0:", rules[0])
