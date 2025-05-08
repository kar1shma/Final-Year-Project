"""
generator.py: parametric Horn-clause benchmark (deduction, depth >= 2)
Enhanced with reasoning type classification and improved prompts
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import random
import clingo
from llama_cpp import Llama
import networkx as nx
import matplotlib.pyplot as plt

# PRIMITIVES
class Term:
    def __init__(self, name: str, is_variable: bool): 
        self.name = name
        self.is_variable = name
        is_variable

    def __repr__(self): 
        return self.name

class Predicate:
    def __init__(self, name:str, arity:int, arg_types:Optional[List[str]]=None, nl_template:str|None=None):
        self.name = name
        self.arity = arity
        self.arg_types   = arg_types or ["any"]*arity
        self.nl_template = nl_template or self._default_template()

    def _default_template(self):
        if self.arity==0: return f"It is {self.name}"
        if self.arity==1: return f"{{0}} is {self.name}"
        return f"{self.name}({', '.join(f'{{{i}}}' for i in range(self.arity))})"
    
    def __repr__(self):
        return f"{self.name}/{self.arity}"

class Atom:
    def __init__(self, predicate:Predicate, terms:List[Term]):
        assert len(terms) == predicate.arity  # ensure arity matches
        self.predicate = predicate
        self.terms = terms

    def __repr__(self):
        if self.predicate.arity==0: return self.predicate.name
        return f"{self.predicate.name}({','.join(t.name for t in self.terms)})"
    
    def to_asp(self): 
        return f"{self}."
    
    def to_nl(self):  
        return self.predicate.nl_template.format(*[t.name for t in self.terms])

class Rule:
    def __init__(self, head:Atom, body:Optional[List[Atom]]=None):
        self.head = head
        self.body = body or []

    def is_fact(self): 
        return not self.body
    
    def __repr__(self):
        return self.head.to_asp()[:-1] if self.is_fact() else f"{self.head.to_asp()[:-1]} :- {', '.join(map(str,self.body))}."
    
    def to_asp(self):
        return self.head.to_asp() if self.is_fact() else f"{self.head} :- {', '.join(map(str,self.body))}."

class LogicProgram(list[Rule]):
    def __init__(self, rules:Optional[List[Rule]]=None):
        super().__init__(rules or [])
        self.rules = rules or []
    
    def __repr__(self):
        return "\n".join(map(str,self.rules))
    
    def to_asp(self): 
        return "\n".join(map(str,self.rules))

# PREDICATES
PREDICATE_POOL=[
    Predicate("sunny",0,[],           "It is sunny"),
    Predicate("wet",1,  ["object"],   "{0} is wet"),
    Predicate("big",1,  ["object"],   "{0} is big"),
    Predicate("sad",1,  ["person"],   "{0} is sad"),
    Predicate("tall",1, ["person"],   "{0} is tall"),
    Predicate("happy",1,["person"],   "{0} is happy"),
    Predicate("person",1,["person"],  "{0} is a person"),
    Predicate("owns",2, ["person","object"], "{0} owns {1}"),
    Predicate("likes",2, ["person","entity"], "{0} likes {1}"),
    Predicate("dislikes",2, ["person","entity"], "{0} dislikes {1}"),
]

# CONSTANTS
CONSTANT_POOL={
    "person":["alice","bob","carol","dave"],
    "object":["apple","book","ball","car"],
    "entity":["alice","bob","carol","dave","apple","book","ball","car"]
}

# CONFIGURATION
CONFIG=dict(num_rules=10,max_body_length=2,allow_recursion=True,branching_factor=2)

# VARIABLES
VAR_NAMES=list("XYZWABC")

# ───────────────────────────────────────────── random generation & grounding
def _new_var(used:set[str]): 
    return next(v for v in VAR_NAMES if v not in used)

def _gen_terms(pred:Predicate, pool:list[Term]):
    used={t.name for t in pool}
    out=[]
    for _ in range(pred.arity):
        v=_new_var(used); used.add(v); t=Term(v,True); pool.append(t)
        out.append(t)
    return out, pool

def _gen_rule(pools,config,head_pred):
    head_ts,pool=_gen_terms(head_pred,[])
    body=[]
    for _ in range(random.randint(1, config["max_body_length"])):
        p = head_pred if config["allow_recursion"] and random.random()<.25 else random.choice(pools)
        ts, pool=_gen_terms(p,pool); body.append(Atom(p, ts))
    dangling = {t.name for t in head_ts}-{t.name for a in body for t in a.terms}
    for v in dangling:                                # bind each head‑only var
        p = random.choice([q for q in pools if q.arity==1 and q is not head_pred])
        t = next(t for t in pool if t.name==v)
        body.append(Atom(p, [t]))
    return Rule(Atom(head_pred, head_ts),body)

def generate_logic_program(config=CONFIG):
    rules = []
    used = {}
    for _ in range(config["num_rules"]):
        if used and random.random()<.5:
            cand = [p for p, c in used.items() if c<config["branching_factor"]]
            hpred = random.choice(cand) if cand else random.choice(PREDICATE_POOL)
        else: hpred = random.choice(PREDICATE_POOL)
        r=_gen_rule(PREDICATE_POOL, config, hpred)
        rules.append(r)
        used[hpred] = used.get(hpred,0) + 1
    return LogicProgram(rules)

# --- grounding
def _instantiate(rule:Rule,pool):
    sub = {}
    pick=lambda tp: random.choice(sum(pool.values(),[])) if tp=="any" else random.choice(pool[tp])
    def inst(a:Atom):
        ts=[]
        for i,t in enumerate(a.terms):
            if t.name not in sub: sub[t.name]=pick(a.predicate.arg_types[i])
            ts.append(Term(sub[t.name], False))
        return Atom(a.predicate,ts)
    return Rule(inst(rule.head),[inst(b) for b in rule.body])

def ground(lp,pool):
    return LogicProgram([_instantiate(r,pool) for r in lp])

# ───────────────────────────────────────────── clingo helper
def synthesise_world(rules_src:str,facts_src:str):
    ctl=clingo.Control()
    ctl.add("base", [], rules_src+"\n" + facts_src)
    ctl.ground([("base", [])]); out = []
    ctl.solve(on_model = lambda m: out.extend(map(str, m.symbols(shown=True))))
    return out

# ──────────────────────────────────────────── proof‑depth utilities
def ground_atom(a:Atom)->str: return f"{a}."

def normalise_atom(atom_str):
    """Remove trailing full-stop from atom strings for consistent comparison."""
    return atom_str[:-1] if atom_str.endswith('.') else atom_str

def min_proof_depth(q:Atom, lp:LogicProgram, facts:list[str]):
    """Calculate minimum proof depth with normalised atom strings."""
    # normalise all facts
    known = {normalise_atom(fact) for fact in facts}
    depth = 0
    
    while True:
        # Check if query is already known
        if normalise_atom(ground_atom(q)) in known:
            return depth
            
        # Derive new facts
        new = set()
        for r in lp:
            # Check if all body atoms are known
            if all(normalise_atom(ground_atom(b)) in known for b in r.body):
                head = normalise_atom(ground_atom(r.head))
                new.add(head)
                
        # If no new facts, query is not derivable
        if not new - known:
            raise ValueError("query not derivable")
            
        # Add new facts and increment depth
        known |= new
        depth += 1

def build_seed_facts(lp:LogicProgram,min_depth:int=2) -> Tuple[list[str],Atom]:
    fact_list:list[str]=[]
    fact_set:set[str]=set()
    candidates=[r for r in lp if not r.is_fact() and r.head.predicate.arity > 0]
    random.shuffle(candidates)

    for r in candidates:
        try:
            if min_proof_depth(r.head,lp,fact_list) >= min_depth:
                return fact_list, r.head
        except ValueError: pass

        # still not deep enough – add 1 body atom that isn't already a fact
        for b in r.body:
            ga=ground_atom(b)
            if ga not in fact_set:
                fact_list.append(ga); fact_set.add(ga)
            break   # add at most 1 per loop

    # fallback: last candidate
    tgt = candidates[-1]
    for b in tgt.body:
        ga=ground_atom(b)
        if ga not in fact_set:
            fact_list.append(ga); fact_set.add(ga)
    return fact_list, tgt.head

# ───────────────────────────────────────────── new func to simulate Clingo reasoning
def explain_reasoning_steps(rules, facts, query, proof_steps):
    """
    Generate a natural language explanation of the reasoning steps.
    """
    explanation = ["Reasoning process:"]
    
    # Start with known facts
    explanation.append("Starting with known facts:")
    for fact in facts:
        explanation.append(f"- {fact}")
    
    # Explain each step in the proof
    explanation.append("Steps:")
    for step_num, (derived_fact, rule_idx) in enumerate(proof_steps, 1):
        if rule_idx is not None:
            rule = rules[rule_idx]
            body_conditions = ", ".join(b.to_nl() for b in rule.body)
            explanation.append(f"Step {step_num}: Since {body_conditions}, we can conclude that {rule.head.to_nl()}.")
    
    # Final conclusion
    explanation.append(f"\nTherefore, {query.to_nl()} is {'true' if proof_steps else 'false'}.")
    
    return "\n".join(explanation)


def debug_clingo_reasoning(rules, facts, query):
    """
    Debug function that tracks the reasoning steps for explanation.
    """
    print("\n===== DEBUGGING CLINGO REASONING =====")
    print("Rules:")
    for i, rule in enumerate(rules):
        print(f"R{i+1}: {rule}")
    
    print("\nFacts:")
    for fact in facts:
        print(f"  {fact}")
    
    print("\nQuery: Does", query.to_nl(), "hold?")
    
    # Simulate the computation that Clingo performs
    known = set(facts)
    iterations = 0
    proof_steps = []  # Track (derived_fact, rule_idx) pairs
    
    print("\nDeduction process:")
    while True:
        iterations += 1
        print(f"\nIteration {iterations}:")
        
        # Compute new facts derivable in this iteration
        new_facts = set()
        for i, rule in enumerate(rules):
            # Skip facts (rules with empty body)
            if rule.is_fact():
                continue
                
            # Check if all body atoms are known
            body_satisfied = True
            for body_atom in rule.body:
                body_atom_str = ground_atom(body_atom)
                if normalise_atom(body_atom_str) not in {normalise_atom(f) for f in known}:
                    body_satisfied = False
                    break
            
            # If body is satisfied, derive the head
            if body_satisfied:
                head_str = ground_atom(rule.head)
                normalised_head = normalise_atom(head_str)
                if normalised_head not in {normalise_atom(f) for f in known}:
                    new_facts.add(head_str)
                    print(f"  Applied R{i+1} to derive: {head_str}")
                    proof_steps.append((head_str, i))  # Store the derived fact and rule index
        
        # If no new facts, we've reached a fixed point
        if not new_facts:
            break
            
        # Add new facts to known facts
        known.update(new_facts)
    
    # Check if query is derivable
    query_str = ground_atom(query)
    normalised_query = normalise_atom(query_str)
    normalised_known = {normalise_atom(f) for f in known}
    result = normalised_query in normalised_known
    
    print("\nFinal result:")
    print(f"Query '{query_str}' is {'derivable' if result else 'not derivable'}")
    print(f"Answer: {'YES' if result else 'NO'}")
    
    return result, proof_steps

# ───────────────────────────────────────────── Prompt for LLM
def generate_explanation_prompt(rules, facts, query):
    """
    Generate a prompt that asks the LLM to explain its reasoning process.
    
    Args:
        rules: List of Rule objects
        facts: List of fact strings
        query: The query Atom object
        
    Returns:
        str: Explanation prompt
    """
    # Convert rules to natural language with numbers for reference
    nl_rules = []
    for i, r in enumerate(rules, 1):
        if r.is_fact():
            nl_rules.append(f"{r.head.to_nl()}.")
        else:
            conditions = ", ".join(b.to_nl() for b in r.body)
            nl_rules.append(f"If {conditions}, then {r.head.to_nl()}.")
    
    # Format facts
    nl_facts = [f"* {normalise_atom(f)}." for f in facts]
    
    prompt = f"""
You are a logical reasoning assistant. Given some information, you must determine if a query follows.

BACKGROUND KNOWLEDGE:
{chr(10).join(nl_rules)}

KNOWN INFORMATION:
{chr(10).join(nl_facts)}

QUESTION:
Does {query.to_nl()} hold?

Approach this step by step, explaining each step of your reasoning clearly.
After explaining your reasoning, answer with YES or NO.
"""
    
    return prompt


def build_grounded_world_graph(logic_program: LogicProgram):
    """
    Build a directed graph from grounded rules: objects are nodes,
    binary predicates are labeled edges, unary predicates become self-loops.
    """
    G = nx.MultiDiGraph()
    for rule in logic_program.rules:
        for atom in [rule.head] + rule.body:
            pred = atom.predicate
            if pred.arity == 2:
                src, tgt = atom.terms[0].name, atom.terms[1].name
                G.add_edge(src, tgt, label=pred.name)
            elif pred.arity == 1:
                node = atom.terms[0].name
                G.add_edge(node, node, label=pred.name)
    return G

def draw_grounded_world_graph(G):
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=400)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
    plt.title("Grounded World Graph")
    plt.show()

def build_dependency_graph(logic_program: LogicProgram):
    """
    Create a predicate-level dependency graph: head -> body edges.
    """
    G = nx.DiGraph()
    for rule in logic_program.rules:
        head = rule.head.predicate.name
        for atom in rule.body:
            G.add_edge(head, atom.predicate.name)
    return G

def draw_dependency_graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=600, arrows=True)
    plt.title("Predicate Dependency Graph")
    plt.show()


# ───────────────────────────────────────────── LLM helper
llm=Llama(model_path="models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",n_ctx=2048, verbose=False)

def ask_llm_yesno(prompt:str)->bool:
    txt=llm(prompt, max_tokens=1024)["choices"][0]["text"]
    # Look for YES/NO at the end of the text
    lines = txt.strip().split('\n')
    last_lines = " ".join(lines[-3:]).lower()  # Look at last 3 lines for conclusion
    return "yes" in last_lines and not ("no" in last_lines and last_lines.rfind("no") > last_lines.rfind("yes"))

MAX_ATTEMPTS=1000

def main():
    for _ in range(MAX_ATTEMPTS):
        gp_abstract = generate_logic_program()
        gp_grounded = ground(gp_abstract, CONSTANT_POOL)
        try:
            seed, query = build_seed_facts(gp_grounded, min_depth=2)
            depth = min_proof_depth(query, gp_grounded, seed)
            if depth>=2: 
                break
        except ValueError:
            continue
    else:
        raise RuntimeError("depth>=2 sample not found")
    
    dep_G = build_dependency_graph(gp_abstract)
    draw_dependency_graph(dep_G)

    # Get reasoning steps and result
    result, proof_steps = debug_clingo_reasoning(gp_grounded, seed, query)
        
    # Generate explanation of the reasoning process
    explanation = explain_reasoning_steps(gp_grounded, seed, query, proof_steps)
    print(f"\n{explanation}")

    # Check ASP world answer (using normalised comparison)
    asp_world = synthesise_world("\n".join(r.to_asp() for r in gp_grounded), "\n".join(seed))
    query_atom = normalise_atom(ground_atom(query))
    asp_world_normalised = {normalise_atom(fact) for fact in asp_world}
    asp_truth = query_atom in asp_world_normalised

    # Visualize grounded world graph (ASP-entailed atoms)
    asp_atoms = []
    for atom_str in asp_world:
        if "(" in atom_str:
            name, rest = atom_str.split("(",1)
            args = rest.rstrip(")").split(",")
            asp_atoms.append(Atom(Predicate(name, len(args)), [Term(a, False) for a in args]))
        else:
            asp_atoms.append(Atom(Predicate(atom_str, 0), []))
    grounded_lp = LogicProgram([Rule(a, []) for a in asp_atoms])
    world_G = build_grounded_world_graph(grounded_lp)
    draw_grounded_world_graph(world_G)

    # Generate explanation prompt and get LLM answer with explanation
    explanation_prompt = generate_explanation_prompt(gp_grounded, seed, query)
    llm_truth = ask_llm_yesno(explanation_prompt)
    
    # Get and display the full LLM explanation
    llm_explanation = llm(explanation_prompt, max_tokens=1024)["choices"][0]["text"]

    print("=========== EXAMPLE ===========")
    print(explanation_prompt)
    print("--------------------------------")
    print("Proof depth (gold):", depth)
    print("ASP:", "YES" if asp_truth else "NO")
    print("LLM:", "YES" if llm_truth else "NO")
    print("\nLLM's explanation:")
    print(llm_explanation)

if __name__=="__main__":
    main()