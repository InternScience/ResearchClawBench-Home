
import json, re, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'

ATOM_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s+([^,;]+)')

def parse_atoms(text):
    atoms=[]
    for part in [p.strip() for p in text.split(',') if p.strip()]:
        m=ATOM_RE.fullmatch(part)
        if not m:
            continue
        pred=m.group(1)
        args=tuple(a for a in m.group(2).split() if a)
        atoms.append((pred,args))
    return atoms

def canon(atom):
    pred,args=atom
    return pred + '(' + ','.join(args) + ')'

def load_defs(path):
    lines=path.read_text().splitlines()
    defs={}
    i=0
    while i < len(lines):
        name=lines[i].strip()
        if not name:
            i+=1; continue
        if ' ' in name and not name.startswith(('=',)):
            if i+4 < len(lines):
                params_line = lines[i+1].strip()
                req_line = lines[i+2].strip()
                cons_line = lines[i+3].strip()
                deps_line = lines[i+4].strip()
                constr_name=name.split()[0]
                formals=name.split()[1:]
                required=[]
                if '=' in req_line:
                    rhs=req_line.split('=',1)[1].strip()
                    required=parse_atoms(rhs)
                consequences=[]
                if ':' in cons_line:
                    after=cons_line.split(':',1)[1].strip()
                    consequences=parse_atoms(after)
                defs[constr_name]={
                    'signature': formals,
                    'required': required,
                    'consequences': consequences,
                    'raw_name': name,
                }
                i+=5
                continue
        i+=1
    return defs

def substitute(atom, env):
    pred,args=atom
    return (pred, tuple(env.get(a,a) for a in args))

def parse_rules(path):
    rules=[]
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or '=>' not in line:
            continue
        lhs,rhs=[x.strip() for x in line.split('=>',1)]
        premises=parse_atoms(lhs)
        conclusions=parse_atoms(rhs)
        if premises and conclusions:
            rules.append({'premises':premises,'conclusions':conclusions,'raw':line})
    return rules

def backtrack_match(premises, fact_index, env=None, used=None, pos=0):
    env = {} if env is None else dict(env)
    used = [] if used is None else list(used)
    if pos == len(premises):
        yield env, used
        return
    ppred, pargs = premises[pos]
    for fact in fact_index.get(ppred, []):
        fpred, fargs = fact
        if len(pargs) != len(fargs):
            continue
        local = dict(env)
        ok=True
        for p, f in zip(pargs, fargs):
            if p in local:
                if local[p] != f:
                    ok=False; break
            else:
                local[p]=f
        if ok:
            yield from backtrack_match(premises, fact_index, local, used+[fact], pos+1)

def parse_problem_pairs(path):
    raw=[ln.rstrip('\n') for ln in path.read_text().splitlines()]
    pairs=[]
    for i in range(0,len(raw),2):
        pid=raw[i].strip()
        stmt=raw[i+1].strip()
        pairs.append((pid,stmt))
    return pairs

def parse_problem(stmt):
    pre, goal = stmt.rsplit('?',1)
    constructions=[c.strip() for c in pre.split(';') if c.strip()]
    goal_atoms=parse_atoms(goal.strip())
    return constructions, goal_atoms[0] if goal_atoms else None

def handle_construction(cons, defs):
    left,right=[x.strip() for x in cons.split('=',1)]
    outputs=tuple(x for x in left.split() if x)
    chunks=[c.strip() for c in right.split(';') if c.strip()]
    facts=[]
    expansion_records=[]
    for ch in chunks:
        parts=ch.split()
        head=parts[0]
        args=tuple(parts[1:])
        facts.append((head,args))
        if head in defs:
            formals=defs[head]['signature']
            vals=list(outputs)+list(args)
            env={k:v for k,v in zip(formals, vals)}
            for atom in defs[head]['required']+defs[head]['consequences']:
                facts.append(substitute(atom, env))
            expansion_records.append({'construction': ch, 'expanded_with': head, 'bindings': env})
    return facts, expansion_records

def solve_problem(pid, stmt, defs, rules, max_iters=4):
    t0=time.time()
    constructions, goal=parse_problem(stmt)
    facts=[]
    trace=[]
    for cons in constructions:
        if '=' in cons:
            newfacts, expansions = handle_construction(cons, defs)
            for f in newfacts:
                facts.append(f)
            for ex in expansions:
                trace.append({'type':'definition_expansion','detail':ex})
    fact_set=set(canon(f) for f in facts)
    fact_index=defaultdict(list)
    for f in facts:
        if canon(f) not in [canon(x) for x in fact_index[f[0]]]:
            fact_index[f[0]].append(f)
    applied=0
    for _ in range(max_iters):
        added=False
        for rule in rules:
            for env, used in backtrack_match(rule['premises'], fact_index):
                for concl in rule['conclusions']:
                    cf=substitute(concl, env)
                    key=canon(cf)
                    if key not in fact_set:
                        fact_set.add(key)
                        fact_index[cf[0]].append(cf)
                        trace.append({'type':'rule_application','rule':rule['raw'],'premises':[canon(u) for u in used],'derived':key})
                        applied += 1
                        added=True
        if not added:
            break
    goal_key = canon(goal) if goal else None
    solved = goal_key in fact_set if goal_key else False
    return {
        'problem_id': pid,
        'goal': goal_key,
        'solved': solved,
        'num_initial_facts': len(facts),
        'num_total_facts': len(fact_set),
        'num_trace_steps': len(trace),
        'num_rule_applications': applied,
        'search_time_seconds': round(time.time()-t0, 6),
        'trace_sample': trace[:20],
    }

def main():
    defs=load_defs(DATA/'defs.txt')
    rules=parse_rules(DATA/'rules.txt')
    results=[]
    for pid,stmt in parse_problem_pairs(DATA/'imo_ag_30.txt'):
        results.append(solve_problem(pid, stmt, defs, rules))
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS/'per_problem_results.json').write_text(json.dumps(results, indent=2))
    agg={
        'num_problems': len(results),
        'num_solved': sum(r['solved'] for r in results),
        'solve_rate': sum(r['solved'] for r in results)/len(results) if results else 0.0,
        'mean_initial_facts': sum(r['num_initial_facts'] for r in results)/len(results) if results else 0.0,
        'mean_total_facts': sum(r['num_total_facts'] for r in results)/len(results) if results else 0.0,
        'mean_trace_steps': sum(r['num_trace_steps'] for r in results)/len(results) if results else 0.0,
        'mean_search_time_seconds': sum(r['search_time_seconds'] for r in results)/len(results) if results else 0.0,
    }
    (OUTPUTS/'aggregate_metrics.json').write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))

if __name__ == '__main__':
    main()
