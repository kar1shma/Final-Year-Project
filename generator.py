"""
generator.py: A parametric Horn-clause benchmark generator.
Uses proper grounding, closed-world assumption, proof tracing, and LLM integration.
Supports first-order variable reuse in rule generation.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set, Any
import random
import clingo
from llama_cpp import Llama
import json
import networkx as nx
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import itertools

# ───────────────────────────────────────────── Primitives
VAR_NAMES = list("XYZWABCDE")


class Term:
    def __init__(self, name: str, is_variable: bool):
        self.name = name
        self.is_variable = is_variable

    def __repr__(self):
        return self.name


class Predicate:
    def __init__(
        self,
        name: str,
        arity: int,
        arg_types: Optional[List[str]] = None,
        nl_template: str = None,
    ):
        self.name = name
        self.arity = arity
        self.arg_types = arg_types or ["any"] * arity
        self.nl_template = nl_template or self._default_template()

    def _default_template(self):
        if self.arity == 0:
            return f"it is {self.name}"
        if self.arity == 1:
            return f"{{0}} is {self.name}"
        return f"{self.name}({', '.join(f'{{{i}}}' for i in range(self.arity))})"

    def __repr__(self):
        return f"{self.name}/{self.arity}"


class Atom:
    def __init__(self, predicate: Predicate, terms: List[Term]):
        assert len(terms) == predicate.arity
        self.predicate = predicate
        self.terms = terms

    def __repr__(self):
        if self.predicate.arity == 0:
            return self.predicate.name
        return f"{self.predicate.name}({','.join(t.name for t in self.terms)})"

    def to_asp(self) -> str:
        return f"{self}."

    def to_nl(self) -> str:
        return self.predicate.nl_template.format(*[t.name for t in self.terms])

    def __eq__(self, other):
        return isinstance(other, Atom) and repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class Rule:
    def __init__(self, head: Atom, body: Optional[List[Atom]] = None):
        self.head = head
        self.body = body or []

    def is_fact(self) -> bool:
        return not self.body

    def __repr__(self) -> str:
        return (
            f"{self.head}."
            if self.is_fact()
            else f"{self.head} :- {', '.join(map(str,self.body))}."
        )

    def to_asp(self) -> str:
        return repr(self)

    def to_nl(self) -> str:
        if self.is_fact():
            return f"{self.head.to_nl()}."
        cond = ", ".join(b.to_nl() for b in self.body)
        return f"If {cond}, then {self.head.to_nl()}."


class LogicProgram:
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules = rules or []

    def __iter__(self):
        return iter(self.rules)

    def to_asp(self) -> str:
        return "\n".join(r.to_asp() for r in self.rules)


# ───────────────────────────────────────────── Predicates & Constants
PREDICATE_POOL: List[Predicate] = [
    Predicate("sunny", 0, [], "it is sunny"),
    Predicate("cold", 0, [], "it is cold"),
    Predicate("hot", 0, [], "it is hot"),
    Predicate("wet", 1, ["object"], "{0} is wet"),
    Predicate("big", 1, ["object"], "{0} is big"),
    Predicate("small", 1, ["object"], "{0} is small"),
    Predicate("sad", 1, ["person"], "{0} is sad"),
    Predicate("tall", 1, ["person"], "{0} is tall"),
    Predicate("happy", 1, ["person"], "{0} is happy"),
    Predicate("hungry", 1, ["person"], "{0} is hungry"),
    Predicate("owns", 2, ["person", "object"], "{0} owns {1}"),
    Predicate("likes", 2, ["person", "entity"], "{0} likes {1}"),
    Predicate("dislikes", 2, ["person", "entity"], "{0} dislikes {1}"),
    Predicate("friend", 2, ["person", "person"], "{0} is a friend of {1}"),
    Predicate("enemy", 2, ["person", "person"], "{0} is an enemy of {1}"),
    Predicate("parent", 2, ["person", "person"], "{0} is a parent of {1}"),
    Predicate("sibling", 2, ["person", "person"], "{0} is a sibling of {1}"),
]
CONSTANT_POOL: Dict[str, List[str]] = {
    "person": ["alice", "bob", "carol", "dave", "eve", "frank", "george"],
    "object": ["apple", "book", "ball", "car", "pencil", "phone"],
    # entity = both people and objects
    "entity": ["alice", "bob", "carol", "dave", "eve", "frank", "george", 
               "apple", "book", "ball", "car", "pencil", "phone"]
}

# ───────────────────────────────────────────── Configuration
CONFIG: Dict[str, Any] = {
    "num_rules": 4,
    "max_body_length": 3,
    "allow_recursion": True,
    "branching_factor": 2,
    "min_proof_depth": 1,
    "num_base_facts": 4,
}

# ───────────────────────────────────────────── Generation Helpers


def _new_var(used_names: Set[str]) -> str:
    return next(v for v in VAR_NAMES if v not in used_names)


# choose fresh or reuse existing variable of correct type
def _choose_term(
    arg_type: str, pool: List[Term], reuse_prob: float = 0.5
) -> Tuple[Term, List[Term]]:
    # existing variables matching type
    candidates = [
        t
        for t in pool
        if t.is_variable and (arg_type == "any" or arg_type in t.name or True)
    ]
    if candidates and random.random() < reuse_prob:
        return random.choice(candidates), pool
    # else fresh
    used_names = {t.name for t in pool}
    new_name = _new_var(used_names)
    new_term = Term(new_name, True)
    return new_term, pool + [new_term]


# generate terms with possible reuse
def _gen_terms(pred: Predicate, pool: List[Term]) -> Tuple[List[Term], List[Term]]:
    out = []
    for arg_type in pred.arg_types:
        term, pool = _choose_term(arg_type, pool)
        out.append(term)

    # if this is a binary predicate, never let the two terms coincide
    if pred.arity == 2 and out[0].name == out[1].name:
        # try to reuse any other existing var
        candidates = [t for t in pool if t.is_variable and t.name != out[0].name]
        if candidates:
            out[1] = random.choice(candidates)
        else:
            # force a fresh one
            used = {t.name for t in pool}
            new_name = _new_var(used)
            new_term = Term(new_name, True)
            pool.append(new_term)
            out[1] = new_term

    return out, pool


# generate a rule with shared variables
def _gen_rule(pools: List[Predicate], config: Dict, head_pred: Predicate) -> Rule:
    head_vars, pool = _gen_terms(head_pred, [])
    body: List[Atom] = []
    for _ in range(random.randint(1, config["max_body_length"])):
        p = (
            head_pred
            if config["allow_recursion"] and random.random() < 0.25
            else random.choice(pools)
        )
        ts, pool = _gen_terms(p, pool)
        body.append(Atom(p, ts))

    # bind dangling head vars
    head_names = {t.name for t in head_vars}
    body_names = {t.name for atom in body for t in atom.terms}
    dangling = head_names - body_names
    for v in dangling:
        p = random.choice([q for q in pools if q.arity == 1])
        # reuse variable v
        term = next(t for t in pool if t.name == v)
        body.append(Atom(p, [term]))
    return Rule(Atom(head_pred, head_vars), body)


def generate_logic_program(config: Dict = CONFIG) -> LogicProgram:
    rules, used_names = [], {}
    for _ in range(config["num_rules"]):
        if used_names and random.random() < 0.5:
            cand = [p for p, c in used_names.items() if c < config["branching_factor"]]
            hp = random.choice(cand) if cand else random.choice(PREDICATE_POOL)
        else:
            hp = random.choice(PREDICATE_POOL)
        r = _gen_rule(PREDICATE_POOL, config, hp)
        rules.append(r)
        used_names[hp] = used_names.get(hp, 0) + 1
    return LogicProgram(rules)


# ───────────────────────────────────────────── Grounding & CWA


def get_all_ground_atoms(lp: LogicProgram) -> Set[Atom]:
    preds = {r.head.predicate for r in lp.rules} | {
        b.predicate for r in lp.rules for b in r.body
    }
    atoms = set()
    for p in preds:
        if p.arity == 0:
            atoms.add(Atom(p, []))
        elif p.arity == 1:
            for c in CONSTANT_POOL.get(p.arg_types[0], []):
                atoms.add(Atom(p, [Term(c, False)]))
        else:
            for c1 in CONSTANT_POOL.get(p.arg_types[0], []):
                for c2 in CONSTANT_POOL.get(p.arg_types[1], []):
                    if c1 != c2:  # avoid self-pairs
                        atoms.add(Atom(p, [Term(c1, False), Term(c2, False)]))
    return atoms


def generate_base_facts(lp: LogicProgram, num_facts: int) -> List[Atom]:
    all_atoms = list(get_all_ground_atoms(lp))
    # drop any 2-ary ground atom like p(c,c)
    all_atoms = [
        a for a in all_atoms
        if not (a.predicate.arity == 2 and a.terms[0].name == a.terms[1].name)
    ]
    return random.sample(all_atoms, k=min(num_facts, len(all_atoms)))


def clingo_grounding(lp: LogicProgram, base_facts: List[Atom]) -> Dict[str, Set[Atom]]:
    rules = lp.to_asp()
    facts = "\n".join(a.to_asp() for a in base_facts)
    preds = {r.head.predicate for r in lp.rules} | {
        b.predicate for r in lp.rules for b in r.body
    }
    shows = "\n".join(f"#show {p.name}/{p.arity}." for p in preds)
    ctl = clingo.Control(arguments = ["--warn=none"])
    # ctl = clingo.Control()
    ctl.add("base", [], rules + "\n" + facts + "\n" + shows)
    ctl.ground([("base", [])])
    true_strs = set()
    ctl.solve(
        on_model=lambda m: true_strs.update(str(x) for x in m.symbols(shown=True))
    )
    true_atoms = set()
    for s in true_strs:
        if "(" in s:
            name, args = s.split("(", 1)
            args = args.rstrip(")").split(",")
            pred = next(
                (p for p in PREDICATE_POOL if p.name == name and p.arity == len(args)),
                None,
            )
            if pred:
                true_atoms.add(Atom(pred, [Term(a, False) for a in args]))
        else:
            pred = next(
                (p for p in PREDICATE_POOL if p.name == s and p.arity == 0), None
            )
            if pred:
                true_atoms.add(Atom(pred, []))
    true_atoms = {a for a in true_atoms if not (a.predicate.arity == 2 and a.terms[0].name == a.terms[1].name)}
    all_atoms = get_all_ground_atoms(lp)
    false_atoms = all_atoms - true_atoms
    return {"true_facts": true_atoms, "false_facts": false_atoms}


# ───────────────────────────────────────────── Proof Tracing


def _generate_substitutions(vars: Dict[str, str]) -> List[Dict[str, str]]:
    if not vars:
        return [{}]
    name, typ = next(iter(vars.items()))
    rest = vars.copy()
    rest.pop(name)
    subs_rest = _generate_substitutions(rest)
    out = []
    for c in CONSTANT_POOL.get(typ, []):
        for sub in subs_rest:
            d = sub.copy()
            d[name] = c
            out.append(d)
    return out


def _apply_substitution(a: Atom, sub: Dict[str, str]) -> Atom:
    terms = []
    for t in a.terms:
        if t.is_variable:
            terms.append(Term(sub.get(t.name, t.name), False))
        else:
            terms.append(t)
    return Atom(a.predicate, terms)


def _get_rule_groundings(rule: Rule) -> List[Dict[str, Any]]:
    vars = {}
    for atom in [rule.head] + rule.body:
        for i, t in enumerate(atom.terms):
            if t.is_variable:
                vars[t.name] = atom.predicate.arg_types[i]
    subs = _generate_substitutions(vars)
    groundings = []
    for sub in subs:
        h = _apply_substitution(rule.head, sub)
        b = [_apply_substitution(x, sub) for x in rule.body]
        groundings.append({"head": h, "body": b})
    return groundings


def determine_how_proved(
    lp: LogicProgram, base: List[Atom], target: Atom
) -> Dict[str, Any]:
    known = set(base)
    derived = {str(a): 0 for a in base}
    deriv_rules = {}
    steps = []
    itr = 0
    while True:
        itr += 1
        new = set()
        for idx, rule in enumerate(lp.rules):
            if rule.is_fact():
                continue
            for gr in _get_rule_groundings(rule):
                if all(b in known for b in gr["body"]):
                    h = gr["head"]
                    if h not in known:
                        new.add(h)
                        deriv_rules[str(h)] = {
                            "rule_idx": idx,
                            "body_atoms": [str(x) for x in gr["body"]],
                        }
                        derived[str(h)] = itr
        if not new:
            break
        known |= new
    if target not in known:
        return {"derivable": False, "steps": [], "depth": -1}
    cur = str(target)
    depth = derived[cur]
    while cur in deriv_rules:
        info = deriv_rules[cur]
        steps.append({"derived_fact": cur, **info})
        bodies = info["body_atoms"]
        if not bodies:
            break
        cur = max(bodies, key=lambda x: derived.get(x, 0))
    steps.reverse()
    return {"derivable": True, "steps": steps, "depth": depth}


# ───────────────────────────────────────────── Target Atom Selection

def select_target(
    all_facts: Dict[str, Set[Atom]],
    base: List[Atom],
    grounded: Dict[str, Dict[str, Any]],
    min_depth: int,
    yes_prob: float = 0.5,
) -> Tuple[Atom, bool]:
    """
    Balanced YES/NO selection. 50% of the time we pick a derivable target
    (weighted by proof depth), otherwise a false one.  When YES, only atoms
    with proof_depth >= min_depth (or >=1 if none meet min_depth).
    """

    base_set = set(base)

    # 1) collect all derivable (non-base) atoms with their depths
    derivables = [
        (atom, info["depth"])
        for atom in all_facts["true_facts"]
        if (info := grounded.get(str(atom), {})).get("derivable", False)
        and atom not in base_set
    ]

    # 2) collect all false atoms
    falses = list(all_facts["false_facts"])

    # 3) decide if we want a YES
    want_yes = bool(derivables) and (random.random() < yes_prob)

    if want_yes:
        # filter by min_depth; if empty, fall back to depth>=1
        candidates = [(a, d) for (a, d) in derivables if d >= min_depth]
        if not candidates:
            candidates = [(a, d) for (a, d) in derivables if d >= 1]

        atoms, depths = zip(*candidates)
        total = sum(depths)
        weights = [d / total for d in depths]
        chosen = random.choices(atoms, weights=weights, k=1)[0]
        return chosen, True

    # NO branch: pick a false atom if possible
    if falses:
        return random.choice(falses), False

    # fallback #1: if there are derivables but no falses, force a YES
    if derivables:
        chosen, _ = random.choice(derivables)
        return chosen, True

    # fallback #2: truly degenerate case — pick any ground atom
    all_atoms = list(all_facts["true_facts"] | all_facts["false_facts"])
    if all_atoms:
        chosen = random.choice(all_atoms)
        return chosen, (chosen in all_facts["true_facts"])

    # shouldn’t happen
    raise RuntimeError(
        "select_target: no atoms at all in true_facts or false_facts!"
    )

# ───────────────────────────────────────────── Explanation
def explain_reasoning_steps(
    lp: LogicProgram,
    base: List[Atom],
    target: Atom,
    info: Dict[str, Any],
    true_facts: Set[Atom],
) -> str:
    if not info.get("derivable", False):
        return f"'{target.to_nl()}' cannot be derived."
    lines = ["Reasoning process:"] + [f"- {a.to_nl()}" for a in base]
    for i, st in enumerate(info["steps"], 1):
        head = next(a for a in true_facts if str(a) == st["derived_fact"])
        body = [next(a for a in true_facts if str(a) == x) for x in st["body_atoms"]]
        cond = ", ".join(b.to_nl() for b in body)
        lines.append(
            f"Step {i}: Since {cond}, by Rule {st['rule_idx']+1} we get {head.to_nl()}."
        )
    lines.append(f"\nTherefore, '{target.to_nl()}' is true.")
    return "\n".join(lines)


# ───────────────────────────────────────────── Visualisation
def build_dependency_graph(lp: LogicProgram) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in lp.rules:
        for b in r.body:
            G.add_edge(r.head.predicate.name, b.predicate.name)
    return G


def draw_dependency_graph(G: nx.DiGraph):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=600)
    plt.title("Predicate Dependency")
    plt.show()


def build_grounded_world_graph(true_facts: Set[Atom]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for a in true_facts:
        if a.predicate.arity == 2:
            G.add_edge(a.terms[0].name, a.terms[1].name, label=a.predicate.name)
        elif a.predicate.arity == 1:
            G.add_edge(a.terms[0].name, a.terms[0].name, label=a.predicate.name)
        else:
            G.add_node(a.predicate.name)
    return G


def draw_grounded_world_graph(G: nx.MultiDiGraph):
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=400)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
    plt.title("Grounded World")
    plt.show()


# ───────────────────────────────────────────── LLM Integration
def setup_llm(
    path: str = "models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
) -> Optional[Llama]:
    try:
        return Llama(model_path=path, n_ctx=10000, verbose=False)
    except:
        print("Warning: failed to load LLM")
        return None


def evaluate_with_llm(llm: Llama, prompt: str) -> Tuple[str, bool]:
    resp = llm(prompt, max_tokens=1024)["choices"][0]["text"].strip()
    verdict = " ".join(resp.split()[-3:]).lower()
    ans = "yes" in verdict and not (
        "no" in verdict and verdict.rfind("no") > verdict.rfind("yes")
    )
    return resp, ans

# ───────────────────────────────────────────── Deduction
def generate_deduction_prompt(
    lp: LogicProgram, base: List[Atom], target: Atom
) -> str:
    ru = [f"Rule {i+1}: {r.to_nl()}" for i, r in enumerate(lp.rules)]
    fa = [f"- {a.to_nl()}" for a in base]
    return f"""
You are given the following information:
{chr(10).join(ru)}

And the following facts:
{chr(10).join(fa)}

QUESTION:
Is “{target.to_nl()}” true?

Answer only with a single word: “YES” or “NO”.
"""

def generate_deduction_task(config: Dict[str,Any]) -> Tuple[str,str,str]:
    # 1) generate program + base facts
    lp         = generate_logic_program(config)
    base       = generate_base_facts(lp, config["num_base_facts"])
    # 2) ground + collect true/false
    all_facts  = clingo_grounding(lp, base)
    # 3) trace proofs, pick target
    proofs     = { str(f): determine_how_proved(lp, base, f)
                   for f in all_facts["true_facts"] }
    target, is_true = select_target(all_facts, base, proofs, config["min_proof_depth"])
    # 4) form q, c, t
    if is_true:
        depth = proofs[str(target)]["depth"]
    else:
        depth = "not applicable"
    q = target.to_asp().strip()
    c = lp.to_asp() + "\n" + "\n".join(f.to_asp() for f in base)
    t = "YES" if is_true else "NO"
    return q, c, t, depth


def build_deduction_benchmark(n: int, config: Dict[str,Any], out_path: str):
    """
    Generate n deduction tasks and write them as JSON to out_path.
    Each task is a dict { "q": ..., "c": ..., "t": ... }.
    """
    bench = []
    for _ in range(n):
        q, c, t, depth = generate_deduction_task(config)
        bench.append({"q": q, "c": c, "t": t, "depth": depth})
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)

# ───────────────────────────────────────────── Abduction
def extract_abduction_hypotheses(proof_info: dict) -> List[str]:
    """
    Given proof_info from determine_how_proved, return the set
    of leaf-premise strings (no trailing dot) that were used to derive the target.
    """
    if not proof_info.get("derivable", False):
        return []
    derived = {step["derived_fact"] for step in proof_info["steps"]}
    leaves = set()
    for step in proof_info["steps"]:
        for atom_str in step["body_atoms"]:
            if atom_str not in derived:
                leaves.add(atom_str)
    return list(leaves)


def generate_abduction_prompt(
    lp: LogicProgram,
    context_facts: List[Atom],
    query: Atom
) -> str:
    """
    Build the pure-NL YES/NO prompt:
      - rules (in NL),
      - context_facts (in NL),
      - QUESTION: Could “<query>” be true?
    """
    rules_nl = "\n".join(f"Rule {i+1}: {r.to_nl()}"
                          for i,r in enumerate(lp.rules))
    facts_nl = "\n".join(f"- {a.to_nl()}." for a in context_facts)
    return f"""
You are given the following rules:
{rules_nl}

And the following facts:
{facts_nl}

QUESTION:
Could “{query.to_nl()}” be true?

Answer only “YES” or “NO”.
""".strip()


def _make_abduction_yes_task(config: Dict[str,Any]) -> Tuple[str,str,str,str]:
    """
    Force a YES-case:
      • pick a derivable target `obs`
      • hide exactly one *leaf* premise needed to prove it
      • context includes all other base–facts + `obs`
      • query is the hidden premise
      • t = "YES"
    """
    for _ in range(200):
        lp   = generate_logic_program(config)
        base = generate_base_facts(lp, config["num_base_facts"])
        all_facts = clingo_grounding(lp, base)
        derivable = [f for f in all_facts["true_facts"] if f not in set(base)]
        if not derivable:
            continue

        obs = random.choice(derivable)
        proof = determine_how_proved(lp, base, obs)
        leaves = extract_abduction_hypotheses(proof)
        if not leaves:
            continue

        # hide one leaf ⇒ that becomes our query
        hidden_str = random.choice(leaves)
        hidden_atom = next(a for a in base if a.to_asp().strip()==hidden_str)

        # rebuild context = (base ∪ {obs}) \ {hidden_atom}
        ctx = [a for a in base if a != hidden_atom] + [obs]

        q_asp   = hidden_atom.to_asp()
        c_asp   = lp.to_asp() + "\n" + "\n".join(a.to_asp() for a in ctx)
        prompt  = generate_abduction_prompt(lp, ctx, hidden_atom)
        return q_asp, c_asp, prompt, "YES"

    raise RuntimeError("Could not generate a YES abduction task")


def _make_abduction_no_task(config: Dict[str,Any]) -> Tuple[str,str,str,str]:
    """
    Force a NO-case:
      • pick a derivable target `obs`
      • hide a fact *not* among its leaf premises
      • context includes all other base–facts + `obs`
      • query is the hidden, irrelevant fact
      • t = "NO"
    """
    for _ in range(200):
        lp   = generate_logic_program(config)
        base = generate_base_facts(lp, config["num_base_facts"])
        all_facts = clingo_grounding(lp, base)
        derivable = [f for f in all_facts["true_facts"] if f not in set(base)]
        if not derivable:
            continue

        obs = random.choice(derivable)
        proof = determine_how_proved(lp, base, obs)
        leaves = set(extract_abduction_hypotheses(proof))

        ctx = base + [obs]
        non_leaves = [a for a in ctx if a.to_asp().strip() not in leaves]
        if not non_leaves:
            continue

        hidden_atom = random.choice(non_leaves)
        ctx_minus    = [a for a in ctx if a != hidden_atom]

        q_asp   = hidden_atom.to_asp()
        c_asp   = lp.to_asp() + "\n" + "\n".join(a.to_asp() for a in ctx_minus)
        prompt  = generate_abduction_prompt(lp, ctx_minus, hidden_atom)
        return q_asp, c_asp, prompt, "NO"

    raise RuntimeError("Could not generate a NO abduction task")


def generate_abduction_task(
    config: Dict[str,Any],
    yes_prob: float = 0.5
) -> Tuple[str,str,str,str]:
    """
    With probability yes_prob build a YES case, otherwise a NO case.
    """
    if random.random() < yes_prob:
        return _make_abduction_yes_task(config)
    else:
        return _make_abduction_no_task(config)


def build_abduction_benchmark(
    n: int,
    config: Dict[str,Any],
    out_path: str,
    yes_prob: float = 0.5
):
    """
    Generate *n* abduction tasks by sampling YES/NO according to yes_prob,
    rather than enforcing an exact split.
    """
    tasks = []
    for _ in range(n):
        q, c, prompt, t = generate_abduction_task(config, yes_prob=yes_prob)
        tasks.append({"q":q, "c":c, "prompt":prompt, "t":t})
    random.shuffle(tasks)
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {len(tasks)} abduction tasks to {out_path}")


if __name__ == "__main__":
    n = 500
    build_deduction_benchmark(n, config=CONFIG, out_path="deduction_benchmark_1.json")
    print(f"Wrote {n} deduction examples to deduction_benchmark_1.json")

    build_abduction_benchmark(n, config=CONFIG, out_path="abduction_benchmark.json")
    print(f"Wrote {n} abduction examples to abduction_benchmark.json")