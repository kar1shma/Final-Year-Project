import logging
import random
import json
from typing import List, Tuple, Optional, Dict, Set, Any, Union
import clingo
from llama_cpp import Llama
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

# ───────────────────────────────────────────── Constants & Logging

VAR_NAMES: List[str] = list("XYZWABCDE")
RETRY_LIMIT: int = 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────── Primitives

class Term:
    def __init__(self, name: str, is_variable: bool):
        self.name = name
        self.is_variable = is_variable

    def __repr__(self) -> str:
        return self.name


class Predicate:
    def __init__(
        self,
        name: str,
        arity: int,
        arg_types: Optional[List[str]] = None,
        nl_template: Optional[str] = None,
    ):
        self.name = name
        self.arity = arity
        self.arg_types = arg_types or ["any"] * arity
        self.nl_template = nl_template or self._default_template()

    def _default_template(self) -> str:
        if self.arity == 0:
            return f"it is {self.name}"
        if self.arity == 1:
            return f"{{0}} is {self.name}"
        return f"{self.name}({', '.join(f'{{{i}}}' for i in range(self.arity))})"

    def __repr__(self) -> str:
        return f"{self.name}/{self.arity}"


class Atom:
    def __init__(self, predicate: Predicate, terms: List[Term]):
        assert len(terms) == predicate.arity
        self.predicate = predicate
        self.terms = terms

    def __repr__(self) -> str:
        if self.predicate.arity == 0:
            return self.predicate.name
        return f"{self.predicate.name}({','.join(t.name for t in self.terms)})"

    def to_asp(self) -> str:
        return f"{self}."

    def to_nl(self) -> str:
        return self.predicate.nl_template.format(*(t.name for t in self.terms))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and repr(self) == repr(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Rule:
    def __init__(self, head: Atom, body: Optional[List[Atom]] = None):
        self.head = head
        self.body = body or []

    def is_fact(self) -> bool:
        return not self.body

    def __repr__(self) -> str:
        if self.is_fact():
            return f"{self.head}."
        return f"{self.head} :- {', '.join(map(str, self.body))}."

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
    "entity": [
        "alice", "bob", "carol", "dave", "eve", "frank", "george",
        "apple", "book", "ball", "car", "pencil", "phone"
    ],
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


def _choose_term(
    arg_type: str,
    pool: List[Term],
    reuse_prob: float = 0.5
) -> Tuple[Term, List[Term]]:
    candidates = [
        t for t in pool
        if t.is_variable and (arg_type == "any" or arg_type in t.name)
    ]
    if candidates and random.random() < reuse_prob:
        return random.choice(candidates), pool
    used_names = {t.name for t in pool}
    new_name = _new_var(used_names)
    new_term = Term(new_name, True)
    return new_term, pool + [new_term]


def _gen_terms(pred: Predicate, pool: List[Term]) -> Tuple[List[Term], List[Term]]:
    out: List[Term] = []
    for arg in pred.arg_types:
        term, pool = _choose_term(arg, pool)
        out.append(term)
    if pred.arity == 2 and out[0].name == out[1].name:
        others = [t for t in pool if t.is_variable and t.name != out[0].name]
        if others:
            out[1] = random.choice(others)
        else:
            used = {t.name for t in pool}
            new_name = _new_var(used)
            new_term = Term(new_name, True)
            pool.append(new_term)
            out[1] = new_term
    return out, pool


def _gen_rule(
    pools: List[Predicate],
    config: Dict[str, Any],
    head_pred: Predicate
) -> Rule:
    head_vars, pool = _gen_terms(head_pred, [])
    body: List[Atom] = []
    for _ in range(random.randint(1, config["max_body_length"])):
        pred = (
            head_pred
            if config["allow_recursion"] and random.random() < 0.25
            else random.choice(pools)
        )
        terms, pool = _gen_terms(pred, pool)
        body.append(Atom(pred, terms))

    head_names = {t.name for t in head_vars}
    body_names = {t.name for atom in body for t in atom.terms}
    dangling = head_names - body_names
    for var in dangling:
        unit_pred = random.choice([p for p in pools if p.arity == 1])
        term = next(t for t in pool if t.name == var)
        body.append(Atom(unit_pred, [term]))
    return Rule(Atom(head_pred, head_vars), body)


def generate_logic_program(config: Dict[str, Any]) -> LogicProgram:
    """
    Generate a random LogicProgram based on the given configuration.
    :param config: parameters controlling rule count, recursion, etc.
    :return: LogicProgram instance.
    """
    rules: List[Rule] = []
    usage: Dict[Predicate, int] = {}
    for _ in range(config["num_rules"]):
        if usage and random.random() < 0.5:
            candidates = [p for p, c in usage.items() if c < config["branching_factor"]]
            head_pred = random.choice(candidates) if candidates else random.choice(PREDICATE_POOL)
        else:
            head_pred = random.choice(PREDICATE_POOL)
        rule = _gen_rule(PREDICATE_POOL, config, head_pred)
        rules.append(rule)
        usage[head_pred] = usage.get(head_pred, 0) + 1
    return LogicProgram(rules)


def get_all_ground_atoms(program: LogicProgram) -> Set[Atom]:
    """
    Enumerate every possible ground Atom for the predicates in the program.
    :param program: a LogicProgram
    :return: set of all ground Atom instances.
    """
    preds = {r.head.predicate for r in program.rules} | {
        b.predicate for r in program.rules for b in r.body
    }
    atoms: Set[Atom] = set()
    for p in preds:
        if p.arity == 0:
            atoms.add(Atom(p, []))
        elif p.arity == 1:
            for c in CONSTANT_POOL.get(p.arg_types[0], []):
                atoms.add(Atom(p, [Term(c, False)]))
        else:
            for c1 in CONSTANT_POOL.get(p.arg_types[0], []):
                for c2 in CONSTANT_POOL.get(p.arg_types[1], []):
                    if c1 != c2:
                        atoms.add(Atom(p, [Term(c1, False), Term(c2, False)]))
    return atoms


def generate_base_facts(program: LogicProgram, num_facts: int) -> List[Atom]:
    """
    Randomly sample ground atoms to serve as base facts.
    :param program: LogicProgram to ground
    :param num_facts: desired count of facts
    :return: list of Atom
    """
    all_atoms = list(get_all_ground_atoms(program))
    filtered = [
        a for a in all_atoms
        if not (a.predicate.arity == 2 and a.terms[0].name == a.terms[1].name)
    ]
    return random.sample(filtered, k=min(num_facts, len(filtered)))


def clingo_grounding(
    program: LogicProgram,
    base_facts: List[Atom]
) -> Dict[str, Set[Atom]]:
    """
    Use clingo to compute which ground atoms hold (true) vs. false under CWA.
    :param program: LogicProgram
    :param base_facts: list of Atom assumed true
    :return: dict with "true_facts" and "false_facts" sets
    """
    asp_rules = program.to_asp()
    asp_facts = "\n".join(a.to_asp() for a in base_facts)
    preds = {r.head.predicate for r in program.rules} | {
        b.predicate for r in program.rules for b in r.body
    }
    shows = "\n".join(f"#show {p.name}/{p.arity}." for p in preds)

    ctl = clingo.Control(arguments=["--warn=none"])
    ctl.add("base", [], asp_rules + "\n" + asp_facts + "\n" + shows)
    ctl.ground([("base", [])])

    true_strs: Set[str] = set()
    try:
        ctl.solve(on_model=lambda m: true_strs.update(str(x) for x in m.symbols(shown=True)))
    except Exception as e:
        logger.warning("Clingo solve failed: %s", e)

    true_atoms: Set[Atom] = set()
    for s in true_strs:
        if "(" in s:
            name, args = s.split("(", 1)
            args = args.rstrip(")").split(",")
            pred = next(
                (p for p in PREDICATE_POOL if p.name == name and p.arity == len(args)),
                None
            )
            if pred:
                true_atoms.add(Atom(pred, [Term(a, False) for a in args]))
        else:
            pred = next(
                (p for p in PREDICATE_POOL if p.name == s and p.arity == 0),
                None
            )
            if pred:
                true_atoms.add(Atom(pred, []))

    true_atoms = {
        a for a in true_atoms
        if not (a.predicate.arity == 2 and a.terms[0].name == a.terms[1].name)
    }
    all_atoms = get_all_ground_atoms(program)
    false_atoms = all_atoms - true_atoms
    return {"true_facts": true_atoms, "false_facts": false_atoms}


def _generate_substitutions(vars_map: Dict[str, str]) -> List[Dict[str, str]]:
    if not vars_map:
        return [{}]
    name, typ = next(iter(vars_map.items()))
    rest = vars_map.copy()
    rest.pop(name)
    subs_rest = _generate_substitutions(rest)
    out: List[Dict[str, str]] = []
    for c in CONSTANT_POOL.get(typ, []):
        for sub in subs_rest:
            new = sub.copy()
            new[name] = c
            out.append(new)
    return out


def _apply_substitution(atom: Atom, sub: Dict[str, str]) -> Atom:
    terms: List[Term] = []
    for t in atom.terms:
        if t.is_variable:
            terms.append(Term(sub.get(t.name, t.name), False))
        else:
            terms.append(t)
    return Atom(atom.predicate, terms)


def _get_rule_groundings(rule: Rule) -> List[Dict[str, Any]]:
    vars_map: Dict[str, str] = {}
    for atom in [rule.head] + rule.body:
        for idx, t in enumerate(atom.terms):
            if t.is_variable:
                vars_map[t.name] = atom.predicate.arg_types[idx]
    subs = _generate_substitutions(vars_map)
    groundings: List[Dict[str, Any]] = []
    for sub in subs:
        head = _apply_substitution(rule.head, sub)
        body = [_apply_substitution(b, sub) for b in rule.body]
        groundings.append({"head": head, "body": body})
    return groundings


def determine_how_proved(
    program: LogicProgram,
    base_facts: List[Atom],
    target: Atom
) -> Dict[str, Any]:
    """
    Trace how (and if) `target` can be derived from `base_facts` via program rules.
    :return: { "derivable": bool, "steps": [...], "depth": int }
    """
    known: Set[Atom] = set(base_facts)
    derived_depth = {str(a): 0 for a in base_facts}
    derivation_rules: Dict[str, Dict[str, Any]] = {}
    iteration = 0

    while True:
        iteration += 1
        new_atoms: Set[Atom] = set()
        for idx, rule in enumerate(program.rules):
            if rule.is_fact():
                continue
            for grounding in _get_rule_groundings(rule):
                if all(b in known for b in grounding["body"]):
                    h = grounding["head"]
                    if h not in known:
                        new_atoms.add(h)
                        derivation_rules[str(h)] = {
                            "rule_idx": idx,
                            "body_atoms": [str(b) for b in grounding["body"]],
                        }
                        derived_depth[str(h)] = iteration
        if not new_atoms:
            break
        known |= new_atoms

    if target not in known:
        return {"derivable": False, "steps": [], "depth": -1}

    # reconstruct one proof path
    path: List[Dict[str, Any]] = []
    current = str(target)
    depth = derived_depth[current]
    while current in derivation_rules:
        info = derivation_rules[current]
        path.append({"derived_fact": current, **info})
        if not info["body_atoms"]:
            break
        current = max(info["body_atoms"], key=lambda x: derived_depth.get(x, 0))
    path.reverse()
    return {"derivable": True, "steps": path, "depth": depth}


def select_target(
    grounded_facts: Dict[str, Set[Atom]],
    base_facts: List[Atom],
    proofs: Dict[str, Dict[str, Any]],
    min_depth: int,
    yes_prob: float = 0.5
) -> Tuple[Atom, bool]:
    """
    Choose a target Atom and whether it should be YES (derivable) or NO.
    :returns: (Atom, is_true)
    """
    base_set = set(base_facts)
    derivables = [
        (atom, info["depth"])
        for atom, info in proofs.items()
        if info.get("derivable", False) and atom not in base_set
    ]
    falses = list(grounded_facts["false_facts"])
    want_yes = bool(derivables) and (random.random() < yes_prob)

    if want_yes:
        candidates = [(a, d) for (a, d) in derivables if d >= min_depth]
        if not candidates:
            candidates = [(a, d) for (a, d) in derivables if d >= 1]
        atoms, depths = zip(*candidates)
        weights = [d / sum(depths) for d in depths]
        return random.choices(atoms, weights=weights, k=1)[0], True

    if falses:
        return random.choice(falses), False

    if derivables:
        return random.choice([a for a, _ in derivables]), True

    all_atoms = list(grounded_facts["true_facts"] | grounded_facts["false_facts"])
    choice = random.choice(all_atoms)
    return choice, (choice in grounded_facts["true_facts"])


def explain_reasoning_steps(
    program: LogicProgram,
    base_facts: List[Atom],
    target: Atom,
    info: Dict[str, Any],
    true_facts: Set[Atom]
) -> str:
    """
    Produce a human-readable trace of the derivation steps.
    """
    if not info.get("derivable", False):
        return f"'{target.to_nl()}' cannot be derived."

    lines = ["Reasoning process:"] + [f"- {a.to_nl()}" for a in base_facts]
    for i, step in enumerate(info["steps"], 1):
        head = next(a for a in true_facts if str(a) == step["derived_fact"])
        body_atoms = [next(a for a in true_facts if str(a) == b) for b in step["body_atoms"]]
        cond = ", ".join(b.to_nl() for b in body_atoms)
        lines.append(
            f"Step {i}: Since {cond}, by Rule {step['rule_idx']+1} we get {head.to_nl()}."
        )
    lines.append(f"\nTherefore, '{target.to_nl()}' is true.")
    return "\n".join(lines)


def build_dependency_graph(program: LogicProgram) -> nx.DiGraph:
    """
    Build a predicate-dependency directed graph from the program rules.
    """
    G = nx.DiGraph()
    for r in program.rules:
        for b in r.body:
            G.add_edge(r.head.predicate.name, b.predicate.name)
    return G


def draw_dependency_graph(G: nx.DiGraph) -> None:
    """
    Display the predicate-dependency graph.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=600)
    plt.title("Predicate Dependency")
    plt.show()


def build_grounded_world_graph(true_facts: Set[Atom]) -> nx.MultiDiGraph:
    """
    Build a grounded-world graph connecting constants by predicates.
    """
    G = nx.MultiDiGraph()
    for a in true_facts:
        if a.predicate.arity == 2:
            G.add_edge(a.terms[0].name, a.terms[1].name, label=a.predicate.name)
        elif a.predicate.arity == 1:
            G.add_edge(a.terms[0].name, a.terms[0].name, label=a.predicate.name)
        else:
            G.add_node(a.predicate.name)
    return G


def draw_grounded_world_graph(G: nx.MultiDiGraph) -> None:
    """
    Display the grounded-world graph.
    """
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=400)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
    plt.title("Grounded World")
    plt.show()


def setup_llm(
    model_path: str = "models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
) -> Optional[Llama]:
    """
    Initialize the Llama LLM from a local model path.
    """
    try:
        return Llama(model_path=model_path, n_ctx=10000, verbose=False)
    except Exception:
        logger.warning("Failed to load LLM from %s", model_path)
        return None


def evaluate_with_llm(llm: Llama, prompt: str) -> Tuple[str, bool]:
    """
    Send `prompt` to the LLM and parse a YES/NO answer.
    """
    resp = llm(prompt, max_tokens=1024)["choices"][0]["text"].strip()
    verdict = " ".join(resp.split()[-3:]).lower()
    answer = "yes" in verdict and not (
        "no" in verdict and verdict.rfind("no") > verdict.rfind("yes")
    )
    return resp, answer


def generate_deduction_prompt(
    program: LogicProgram,
    base_facts: List[Atom],
    target: Atom
) -> str:
    """
    Construct the natural-language prompt for a deduction task.
    """
    rules_nl = [f"Rule {i+1}: {r.to_nl()}" for i, r in enumerate(program.rules)]
    facts_nl = [f"- {a.to_nl()}" for a in base_facts]
    return (
        "You are given the following information:\n"
        + "\n".join(rules_nl)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts_nl)
        + f"\n\nQUESTION:\nIs “{target.to_nl()}” true?\n\nAnswer only with “YES” or “NO”."
    )


def generate_deduction_task(
    config: Dict[str, Any]
) -> Tuple[str, str, str, Union[int, str]]:
    """
    Generate a single (q, c, t, depth) deduction task.
    :return: q = query ASP, c = context ASP, t = “YES”/“NO”, depth of proof or "not applicable".
    """
    program = generate_logic_program(config)
    base_facts = generate_base_facts(program, config["num_base_facts"])
    grounded = clingo_grounding(program, base_facts)
    proofs = {
        str(f): determine_how_proved(program, base_facts, f)
        for f in grounded["true_facts"]
    }
    target, is_true = select_target(grounded, base_facts, proofs, config["min_proof_depth"])
    depth = proofs.get(str(target), {}).get("depth", "not applicable") if is_true else "not applicable"
    q = target.to_asp().strip()
    c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in base_facts)
    t = "YES" if is_true else "NO"
    return q, c, t, depth


def build_deduction_benchmark(
    n: int, config: Dict[str, Any], out_path: str
) -> None:
    """
    Generate n deduction tasks and write them as JSON to out_path.
    """
    tasks = []
    for _ in range(n):
        q, c, t, depth = generate_deduction_task(config)
        tasks.append({"q": q, "c": c, "t": t, "depth": depth})
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)


def extract_abduction_hypotheses(proof_info: Dict[str, Any]) -> List[str]:
    """
    Return the leaf-premise strings used in the proof (no trailing dot).
    """
    if not proof_info.get("derivable", False):
        return []
    derived = {step["derived_fact"] for step in proof_info["steps"]}
    leaves: Set[str] = set()
    for step in proof_info["steps"]:
        for atom_str in step["body_atoms"]:
            if atom_str not in derived:
                leaves.add(atom_str)
    return list(leaves)


def generate_abduction_prompt(
    program: LogicProgram,
    context_facts: List[Atom],
    query: Atom
) -> str:
    """
    Construct the natural-language prompt for an abduction task.
    """
    rules_nl = [f"Rule {i+1}: {r.to_nl()}" for i, r in enumerate(program.rules)]
    facts_nl = [f"- {a.to_nl()}." for a in context_facts]
    return (
        "You are given the following rules:\n"
        + "\n".join(rules_nl)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts_nl)
        + f"\n\nQUESTION:\nCould “{query.to_nl()}” be true?\n\nAnswer only “YES” or “NO”."
    )


def generate_abduction_task(
    config: Dict[str, Any],
    yes_prob: float = 0.5
) -> Tuple[str, str, str, str]:
    """
    Generate a single abduction task (q, c, prompt, t).
    """
    if random.random() < yes_prob:
        return _make_abduction_yes_task(config)
    return _make_abduction_no_task(config)


def _make_abduction_yes_task(config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    for _ in range(RETRY_LIMIT):
        program = generate_logic_program(config)
        base_facts = generate_base_facts(program, config["num_base_facts"])
        grounded = clingo_grounding(program, base_facts)
        candidates = [f for f in grounded["true_facts"] if f not in set(base_facts)]
        if not candidates:
            continue
        obs = random.choice(candidates)
        proof = determine_how_proved(program, base_facts, obs)
        leaves = extract_abduction_hypotheses(proof)
        if not leaves:
            continue
        hidden_str = random.choice(leaves)
        hidden_atom = next(
            a for a in base_facts if a.to_asp().strip().rstrip(".") == hidden_str
        )
        context_facts = [a for a in base_facts if a != hidden_atom] + [obs]
        q = hidden_atom.to_asp()
        c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in context_facts)
        prompt = generate_abduction_prompt(program, context_facts, hidden_atom)
        return q, c, prompt, "YES"
    raise RuntimeError("Could not generate a YES abduction task")


def _make_abduction_no_task(config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    for _ in range(RETRY_LIMIT):
        program = generate_logic_program(config)
        base_facts = generate_base_facts(program, config["num_base_facts"])
        grounded = clingo_grounding(program, base_facts)
        candidates = [f for f in grounded["true_facts"] if f not in set(base_facts)]
        if not candidates:
            continue
        obs = random.choice(candidates)
        proof = determine_how_proved(program, base_facts, obs)
        leaves = set(extract_abduction_hypotheses(proof))
        context_facts = base_facts + [obs]
        non_leaves = [a for a in context_facts if a.to_asp().strip().rstrip(".") not in leaves]
        if not non_leaves:
            continue
        hidden_atom = random.choice(non_leaves)
        context_minus = [a for a in context_facts if a != hidden_atom]
        q = hidden_atom.to_asp()
        c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in context_minus)
        prompt = generate_abduction_prompt(program, context_minus, hidden_atom)
        return q, c, prompt, "NO"
    raise RuntimeError("Could not generate a NO abduction task")


def build_abduction_benchmark(
    n: int,
    config: Dict[str, Any],
    out_path: str,
    yes_prob: float = 0.5
) -> None:
    """
    Generate n abduction tasks and write them as JSON to out_path.
    """
    tasks: List[Dict[str, str]] = []
    for _ in range(n):
        q, c, prompt, t = generate_abduction_task(config, yes_prob=yes_prob)
        tasks.append({"q": q, "c": c, "prompt": prompt, "t": t})
    random.shuffle(tasks)
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    n = 5
    build_deduction_benchmark(n, config=CONFIG, out_path="deduction_benchmark_1.json")
    logger.info("Wrote %d deduction examples to deduction_benchmark_1.json", n)

    build_abduction_benchmark(n, config=CONFIG, out_path="abduction_benchmark.json")
    logger.info("Wrote %d abduction examples to abduction_benchmark.json", n)
