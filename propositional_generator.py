import logging
import random
import json
from typing import List, Tuple, Optional, Dict, Set, Any
import clingo
from itertools import product

# Consts & Logging

RETRY_LIMIT: int = 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predicate:
    """
    A logical predicate symbol with fixed arity and argument‐type constraints.
    - name: e.g. "parent", "wet", "sunny"
    - arity: number of arguments
    - arg_types: for each argument position, which types are allowed
    - nl_template: converts the predicate into a natural language string.
    """

    def __init__(
        self,
        name: str,
        arity: int,
        arg_types: Optional[List[str]] = None,
        nl_template: Optional[str] = None,
    ):
        self.name: str = name
        self.arity: int = arity
        self.arg_types: List[str] = (
            arg_types if arg_types is not None else ["any"] * arity
        )
        self.nl_template: str = (
            nl_template if nl_template is not None else self._default_template()
        )

    def _default_template(self) -> str:
        if self.arity == 0:
            return f"it is {self.name}"
        if self.arity == 1:
            return f"{{0}} is {self.name}"
        placeholders = ", ".join(f"{{{i}}}" for i in range(self.arity))
        return f"{self.name}({placeholders})"

    def __repr__(self) -> str:
        return f"{self.name}/{self.arity}"


class Atom:
    """
    A ground atom (predicate + constant terms).
    - predicate: a Predicate object (with name and arity)
    - terms: a list of constant strings (length must equal predicate.arity)
    """

    def __init__(self, predicate: Predicate, terms: List[str]):
        assert len(terms) == predicate.arity, (
            f"Atom.__init__: predicate {predicate.name}/{predicate.arity} requires exactly "
            f"{predicate.arity} terms, but got {len(terms)}"
        )
        self.predicate: Predicate = predicate
        self.terms: List[str] = terms

    def __repr__(self) -> str:
        if self.predicate.arity == 0:
            return self.predicate.name
        term_str = ",".join(self.terms)
        return f"{self.predicate.name}({term_str})"

    def to_asp(self) -> str:
        return f"{self}."

    def to_nl(self) -> str:
        return self.predicate.nl_template.format(*self.terms)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and repr(self) == repr(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Rule:
    """
    A ground Horn clause with one head (Atom) and zero or more body Atoms.
    - head: an Atom that becomes true if all body Atoms hold.
    - body: a list of Atom objects (empty => a ground fact).
    """

    def __init__(self, head: Atom, body: Optional[List[Atom]] = None):
        self.head: Atom = head
        self.body: List[Atom] = body if body is not None else []

    def is_fact(self) -> bool:
        return len(self.body) == 0

    def __repr__(self) -> str:
        if self.is_fact():
            return f"{self.head}."
        body_str = ", ".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}."

    def to_asp(self) -> str:
        return repr(self)

    def to_nl(self) -> str:
        head_nl = self.head.to_nl()
        if self.is_fact():
            return f"{head_nl}."
        conds_nl = " and ".join(b.to_nl() for b in self.body)
        return f"If {conds_nl}, then {head_nl}."


class LogicProgram:
    """
    A container for a finite set of ground Horn clauses (Rule objects).
    - rules: List[Rule]
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = rules if rules is not None else []

    def __iter__(self):
        return iter(self.rules)

    def to_asp(self) -> str:
        return "\n".join(r.to_asp() for r in self.rules)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)


# Predicates & Constants

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
    Predicate("likes", 2, ["person", "object"], "{0} likes {1}"),
    Predicate("dislikes", 2, ["person", "object"], "{0} dislikes {1}"),
    Predicate("friend", 2, ["person", "person"], "{0} is a friend of {1}"),
    Predicate("enemy", 2, ["person", "person"], "{0} is an enemy of {1}"),
    Predicate("parent", 2, ["person", "person"], "{0} is a parent of {1}"),
    Predicate("sibling", 2, ["person", "person"], "{0} is a sibling of {1}"),
]

CONSTANT_POOL: Dict[str, List[str]] = {
    "person": ["alice", "bob", "carol", "dave", "eve", "frank", "george"],
    "object": ["apple", "book", "ball", "car", "pencil", "phone"],
}


def generate_logic_program(config: Dict[str, Any]) -> LogicProgram:
    """
    Generate a random LogicProgram of ground (variable‐free) Horn rules according to config.
    All terms are constants to ensure safe grounding.
    """
    rules: List[Rule] = []
    usage: Dict[Predicate, int] = {}
    for _ in range(config.get("num_rules", 0)):
        if usage and random.random() < 0.5:
            cands = [
                p for p, c in usage.items() if c < config.get("branching_factor", 1)
            ]
            head_pred = random.choice(cands) if cands else random.choice(PREDICATE_POOL)
        else:
            head_pred = random.choice(PREDICATE_POOL)
        head_terms: List[str] = []
        for typ in head_pred.arg_types:
            choices = CONSTANT_POOL.get(typ, [])
            const = (
                random.choice(choices) if choices else random.choice(sum(CONSTANT_POOL.values(), []))
            )
            head_terms.append(const)
        head = Atom(head_pred, head_terms)
        body: List[Atom] = []
        length = random.randint(1, config.get("max_body_length", 1))
        for _ in range(length):
            pred = (
                head_pred
                if config.get("allow_recursion", False) and random.random() < 0.25
                else random.choice(PREDICATE_POOL)
            )
            terms: List[str] = []
            for typ in pred.arg_types:
                choices = CONSTANT_POOL.get(typ, [])
                const = (
                    random.choice(choices) if choices else random.choice(sum(CONSTANT_POOL.values(), []))
                )
                terms.append(const)
            body.append(Atom(pred, terms))
        rules.append(Rule(head, body))
        usage[head_pred] = usage.get(head_pred, 0) + 1
    return LogicProgram(rules)


def get_all_ground_atoms(program: LogicProgram) -> Set[Atom]:
    """
    Generate all possible ground atoms from the predicates used in a LogicProgram.
    """
    atoms = set()
    preds = {r.head.predicate for r in program.rules} | {
        b.predicate for r in program.rules for b in r.body
    }
    for p in preds:
        if p.arity == 0:
            atoms.add(Atom(p, []))
        elif p.arity == 1:
            for c in CONSTANT_POOL.get(p.arg_types[0], []):
                atoms.add(Atom(p, [c]))
        else:
            for c1 in CONSTANT_POOL.get(p.arg_types[0], []):
                for c2 in CONSTANT_POOL.get(p.arg_types[1], []):
                    if c1 != c2:
                        atoms.add(Atom(p, [c1, c2]))
    return atoms


def generate_base_facts(program: LogicProgram, num_facts: int) -> List[Atom]:
    """
    Generate a random sample of ground atoms from the LogicProgram's predicates.
    """
    pool = list(get_all_ground_atoms(program))
    return random.sample(pool, min(num_facts, len(pool)))


def clingo_grounding(
    program: LogicProgram, base_facts: List[Atom]
) -> Dict[str, Set[Atom]]:
    """
    Ground the LogicProgram using Clingo and return true/false facts.
    """
    asp_rules = program.to_asp()
    asp_facts = "\n".join(a.to_asp() for a in base_facts)
    preds = {r.head.predicate for r in program.rules} | {
        b.predicate for r in program.rules for b in r.body
    }
    shows = "\n".join(f"#show {p.name}/{p.arity}." for p in preds)
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], asp_rules + "\n" + asp_facts + "\n" + shows)
    ctl.ground([("base", [])])
    true_strs = set()
    ctl.solve(
        on_model=lambda m: true_strs.update(str(x) for x in m.symbols(shown=True))
    )
    true_atoms = set()
    for s in true_strs:
        if "(" in s:
            n, args = s.split("(", 1)
            args = args.rstrip(")").split(",")
            p = next(
                (pp for pp in PREDICATE_POOL if pp.name == n and pp.arity == len(args)),
                None,
            )
            if p:
                true_atoms.add(Atom(p, args))
        else:
            p = next(
                (pp for pp in PREDICATE_POOL if pp.name == s and pp.arity == 0), None
            )
            if p:
                true_atoms.add(Atom(p, []))
    all_atoms = get_all_ground_atoms(program)
    return {"true_facts": true_atoms, "false_facts": all_atoms - true_atoms}


def determine_how_proved(
    program: LogicProgram, base_facts: List[Atom], target: Atom
) -> Dict[str, Any]:
    """
    Determine the proof chain for a target atom in the LogicProgram.
    """
    known, set_depth, der_rules = set(base_facts), {str(a): 0 for a in base_facts}, {}
    depth = 0
    while True:
        new = set()
        for i, r in enumerate(program.rules):
            if r.is_fact():
                continue
            if all(b in known for b in r.body):
                h = r.head
                if h not in known:
                    new.add(h)
                    der_rules[str(h)] = {
                        "rule_idx": i,
                        "body_atoms": [str(b) for b in r.body],
                    }
                    set_depth[str(h)] = depth + 1
        if not new:
            break
        depth += 1
        known |= new
    if target not in known:
        return {"derivable": False, "steps": [], "depth": -1}
    steps = []
    cur = str(target)
    while cur in der_rules:
        info = der_rules[cur]
        steps.append({"derived_fact": cur, **info})
        if not info["body_atoms"]:
            break
        cur = max(info["body_atoms"], key=lambda x: set_depth.get(x, 0))
    steps.reverse()
    return {"derivable": True, "steps": steps, "depth": set_depth[str(target)]}


def generate_inductive_program(
    config: Dict[str, Any],
) -> Tuple[LogicProgram, List[Atom]]:
    """
    Inductively build a LogicProgram guaranteed to have a proof chain of length proof_depth with ground atoms.
    """
    d = config.get("proof_depth", 1)
    chain_rules: List[Rule] = []

    prev_head: Optional[Atom] = None
    for i in range(d + 1):
        p = random.choice(PREDICATE_POOL)
        terms: List[str] = []
        for typ in p.arg_types:
            const = random.choice(CONSTANT_POOL.get(typ, []))
            terms.append(const)
        head = Atom(p, terms)
        body = [prev_head] if prev_head is not None else []
        chain_rules.append(Rule(head, body))
        prev_head = head

    remaining = max(config.get("num_rules", 0) - len(chain_rules), 0)
    rnd_cfg = {**config, "num_rules": remaining}

    rest_prog = generate_logic_program(rnd_cfg)
    chain_rule_facts = chain_rules[:1]
    chain_inference = chain_rules[1:]

    program = LogicProgram(chain_inference + rest_prog.rules)

    root_fact = chain_rule_facts[0].head
    other_base = generate_base_facts(
        program, max(config.get("num_base_facts", 1) - 1, 0)
    )
    base_facts = [root_fact] + other_base
    return program, base_facts


def generate_multiple_deduction_tasks(
    program: LogicProgram,
    base: List[Atom],
    config: Dict[str, Any],
    n: int,
    yes_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Build exactly n deduction tasks, half YES and half NO (up to rounding).
    """
    grounded = clingo_grounding(program, base)
    proofs = {
        atom: determine_how_proved(program, base, atom)
        for atom in grounded["true_facts"]
    }
    yes_pool = [a for a, info in proofs.items() if info["derivable"]]
    no_pool = list(grounded["false_facts"])

    num_yes = int(n * yes_ratio)
    num_no = n - num_yes

    yes_samples = random.sample(yes_pool, min(num_yes, len(yes_pool)))
    no_samples = random.sample(no_pool, min(num_no, len(no_pool)))

    tasks = []
    for target in yes_samples + no_samples:
        label = "true" if target in yes_samples else "false"
        depth = config["proof_depth"] if label == "true" else "not applicable"
        q = target.to_asp().strip()
        c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in base)
        nl = generate_deduction_prompt(program, base, target)
        tasks.append(
            {
                "q": q,
                "c": c,
                "natural language": nl,
                "t": label,
                "metadata": {**config, "depth": depth, "reasoning_type": "deduction"},
            }
        )
    return tasks


def extract_abduction_hypotheses(proof_info: Dict[str, Any]) -> List[str]:
    """
    Extract the leaves of the proof tree for abduction tasks.
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


def is_explanation(program: LogicProgram, context: List[Atom], hypothesis: Atom, conclusion: Atom) -> bool:
    """
    Check if adding hypothesis to context makes conclusion derivable, but without hypothesis it's not.
    """
    without = clingo_grounding(program, context)
    if conclusion in without["true_facts"]:
        return False
    with_h = clingo_grounding(program, context + [hypothesis])
    return conclusion in with_h["true_facts"]


def generate_multiple_abduction_tasks(
    program: LogicProgram,
    base: List[Atom],
    config: Dict[str, Any],
    n: int,
    yes_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Generate abductive reasoning tasks: some “true” (h really explains the conclusion)
    and some “false” (h does not explain). We ensure false examples by dropping a
    genuine used base fact and then showing a non‐used fact cannot recover the chain.
    """

    grounded = clingo_grounding(program, base)
    true_facts = list(grounded["true_facts"])

    proof_chains = []
    for atom in true_facts:
        info = determine_how_proved(program, base, atom)
        if info["derivable"] and info["depth"] > 0:
            proof_chains.append((atom, info))

    if not proof_chains:
        return []

    true_tasks = []
    false_tasks = []

    for conclusion, proof_info in proof_chains:
        # 1) Identify which base facts were actually used in the proof of `conclusion`
        used_base = set()
        for step in proof_info["steps"]:
            for body_atom_str in step["body_atoms"]:
                for b in base:
                    if str(b) == body_atom_str:
                        used_base.add(b)

        # 2) Identify which base facts were never used
        non_used = [b for b in base if b not in used_base]

        # === Build “true” abduction tasks ===
        # For each h ∈ used_base, remove h from base and check that:
        #   (a) without h, conclusion is not derivable
        #   (b) with h added back, conclusion is derivable
        for h in used_base:
            if len(true_tasks) + len(false_tasks) >= n:
                break
            context_true = [b for b in base if b != h]
            if not is_explanation(program, context_true, h, conclusion):
                continue
            q = h.to_asp().strip()
            c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in context_true)
            nl = generate_abduction_prompt(program, context_true, h, conclusion)
            true_tasks.append({
                "q": q,
                "c": c,
                "natural language": nl,
                "t": "true",
                "metadata": {
                    **config,
                    "reasoning_type": "abduction",
                    "conclusion": str(conclusion),
                    "explanation": str(h),
                    "proof_depth": proof_info["depth"],
                }
            })

        # === Build “false” abduction tasks ===
        # For each x ∈ used_base, remove x from base so conclusion is broken,
        # then for each h ∈ non_used, check that adding h back STILL fails to derive conclusion.
        for x in used_base:
            if len(true_tasks) + len(false_tasks) >= n:
                break
            context_false_base = [b for b in base if b != x]
            # Confirm conclusion is indeed not derivable without x:
            without_x = clingo_grounding(program, context_false_base)
            if conclusion in without_x["true_facts"]:
                # If removing x does NOT break the chain, skip this x
                continue

            for h in non_used:
                if len(true_tasks) + len(false_tasks) >= n:
                    break
                # Add h to the context where x was removed
                with_h = clingo_grounding(program, context_false_base + [h])
                # If conclusion STILL NOT derivable, h is a “false” explanation
                if conclusion not in with_h["true_facts"]:
                    q = h.to_asp().strip()
                    c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in context_false_base)
                    nl = generate_abduction_prompt(program, context_false_base, h, conclusion)
                    false_tasks.append({
                        "q": q,
                        "c": c,
                        "natural language": nl,
                        "t": "false",
                        "metadata": {
                            **config,
                            "reasoning_type": "abduction",
                            "conclusion": str(conclusion),
                            "non_explanation": str(h),
                            "proof_depth": "not applicable",
                        }
                    })

    # Now sample from both pools to get exactly n tasks
    num_true  = int(n * yes_ratio)
    num_false = n - num_true

    sampled_true  = random.sample(true_tasks,  min(num_true,  len(true_tasks)))
    sampled_false = random.sample(false_tasks, min(num_false, len(false_tasks)))
    return (sampled_true + sampled_false)[:n]


def generate_deduction_prompt(
    program: LogicProgram, base: List[Atom], target: Atom
) -> str:
    """
    Generate a natural language prompt for deduction tasks.
    """
    rules = [r.to_nl() for r in program.rules]
    facts = [f"{a.to_nl()}." for a in base]
    return (
        "You are given the following information:\n"
        + "\n".join(rules)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts)
        + f"\n\nQUESTION:\nIs {target.to_nl()} true?"
        + "\nAnswer exactly true or false."
    )


def generate_abduction_prompt(
    program: LogicProgram, context: List[Atom], hypothesis: Atom, conclusion: Atom
) -> str:
    """
    Generate a natural language prompt for abduction tasks.
    """
    rules = [r.to_nl() for r in program.rules]
    facts = [f"{a.to_nl()}." for a in context]
    return (
        "You are given the following rules:\n"
        + "\n".join(rules)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts)
        + f"\n\nQUESTION:\nCould {hypothesis.to_nl()} explain {conclusion.to_nl()}?"
        + "\nAnswer exactly true or false."
    )


# Main driver: sweeping diverse configs
if __name__ == "__main__":
    NUM_RULES      = [5, 15, 25, 35]
    NUM_BASE_FACTS = [3, 6, 9, 12, 15]
    PROOF_DEPTHS   = [1, 5, 10, 20, 30]
    RECURSION_OPTIONS = [True, False]
    TASKS_PER_GROUP = 5
    GROUPS_PER_CFG  = 1

    all_tasks = []
    for nr, nb, pd, ro in product(NUM_RULES, NUM_BASE_FACTS, PROOF_DEPTHS, RECURSION_OPTIONS):
        if pd > nr:
            continue  # skip impossible configs
        cfg = {
            "num_rules":       nr,
            "max_body_length": 3,
            "allow_recursion": ro,
            "branching_factor": 2,
            "proof_depth":      pd,
            "num_base_facts":   nb,
        }
        for _ in range(GROUPS_PER_CFG):
            prog, base = generate_inductive_program(cfg)
            ded = generate_multiple_deduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
            abd = generate_multiple_abduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
            all_tasks.extend(ded)
            all_tasks.extend(abd)

    with open("propositional_benchmark.json", "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"Generated {len(all_tasks)} total tasks in propositional_benchmark.json")
