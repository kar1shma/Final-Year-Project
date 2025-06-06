import logging
import random
import json
from typing import List, Tuple, Optional, Dict, Set, Any
import clingo
from itertools import product

# --------------------------
# Constants & Logging Setup
# --------------------------
RETRY_LIMIT: int = 200
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
VARIABLE_POOL = ["X", "Y", "Z", "U", "V", "W"]

def _indefinite(typ: str) -> str:
    return "someone" if typ == "person" else "something"


# --------------------------
# Core Classes: Predicate, Atom, Rule, LogicProgram
# --------------------------
class Predicate:
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
    def __init__(self, predicate: Predicate, terms: List[str]):
        assert len(terms) == predicate.arity, (
            f"Predicate {predicate.name}/{predicate.arity} requires {predicate.arity} terms, "
            f"got {len(terms)}"
        )
        self.predicate: Predicate = predicate
        self.terms: List[str] = terms

    def __repr__(self) -> str:
        if self.predicate.arity == 0:
            return self.predicate.name
        return f"{self.predicate.name}({','.join(self.terms)})"

    def to_asp(self) -> str:
        return f"{self}."

    def to_nl(self) -> str:
        words: List[str] = []
        for i, t in enumerate(self.terms):
            if t and t[0].isupper():
                words.append(_indefinite(self.predicate.arg_types[i]))
            else:
                words.append(t)
        return self.predicate.nl_template.format(*words)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and repr(self) == repr(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Rule:
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
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = rules if rules is not None else []

    def __iter__(self):
        return iter(self.rules)

    def to_asp(self) -> str:
        return "\n".join(r.to_asp() for r in self.rules)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)


# --------------------------
# Predicate & Constant Pools
# --------------------------
PREDICATE_POOL: List[Predicate] = [
    Predicate("sunny", 0, [], "it is sunny"),
    Predicate("cold", 0, [], "it is cold"),
    Predicate("hot", 0, [], "it is hot"),
    Predicate("wet", 1, ["object"], "{0} is wet"),
    Predicate("big", 1, ["object"], "{0} is big"),
    Predicate("small", 1, ["object"], "{0} is small"),
    Predicate("clean", 1, ["object"], "{0} is clean"),
    Predicate("red", 1, ["object"], "{0} is red"),
    Predicate("sad", 1, ["person"], "{0} is sad"),
    Predicate("tall", 1, ["person"], "{0} is tall"),
    Predicate("happy", 1, ["person"], "{0} is happy"),
    Predicate("hungry", 1, ["person"], "{0} is hungry"),
    Predicate("calm", 1, ["person"], "{0} is calm"),
    Predicate("excited", 1, ["person"], "{0} is excited"),
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


# --------------------------
# Helper: Compute ALL possible ground atoms
# --------------------------
def get_all_ground_atoms(program: LogicProgram) -> Set[Atom]:
    atoms: Set[Atom] = set()
    preds = {r.head.predicate for r in program.rules} | {
        b.predicate for r in program.rules for b in r.body
    }
    for p in preds:
        if p.arity == 0:
            atoms.add(Atom(p, []))
        elif p.arity == 1:
            for c in CONSTANT_POOL.get(p.arg_types[0], []):
                atoms.add(Atom(p, [c]))
        else:  # arity == 2
            for c1 in CONSTANT_POOL.get(p.arg_types[0], []):
                for c2 in CONSTANT_POOL.get(p.arg_types[1], []):
                    if c1 != c2:
                        atoms.add(Atom(p, [c1, c2]))
    return atoms


def generate_base_facts(program: LogicProgram, num_facts: int) -> List[Atom]:
    pool = list(get_all_ground_atoms(program))
    return random.sample(pool, min(num_facts, len(pool)))


# --------------------------
# Helper: Generate Extra FOL Rules Without Shortcutting the Chain
# --------------------------
def _is_trivial(rule: Rule) -> bool:
    # exactly p(X) :- p(X)
    return (
        len(rule.body)==1
        and rule.head.predicate == rule.body[0].predicate
        and rule.head.terms     == rule.body[0].terms
    )

def _generate_extra_fol_rules(
    num_extra: int,
    max_body_len: int,
    chain_pred_names: Set[str]
) -> List[Rule]:
    extra_rules: List[Rule] = []
    available_preds = [p for p in PREDICATE_POOL if p.name not in chain_pred_names]
    for _ in range(num_extra):
        if not available_preds:
            break
        head_pred = random.choice(available_preds)

        # Build a fresh variable‐to‐type map for the head
        var_map: Dict[str, str] = {}
        head_vars: List[str] = []
        for typ in head_pred.arg_types:
            v = random.choice(VARIABLE_POOL)
            while v in var_map:
                v = random.choice(VARIABLE_POOL)
            var_map[v] = typ
            head_vars.append(v)
        head = Atom(head_pred, head_vars)

        body: List[Atom] = []

        # (1) One body atom per variable in var_map, ensuring no reflexive (X, X) for arity=2
        for v, typ in var_map.items():
            candidates = [p for p in available_preds if typ in p.arg_types and p.arity >= 1]
            if not candidates:
                continue
            p = random.choice(candidates)

            # Build terms for p, placing v in one slot and a different var in the other (if possible)
            terms: List[str] = []
            placed = False
            for arg_type in p.arg_types:
                if arg_type == typ and not placed:
                    terms.append(v)
                    placed = True
                else:
                    other_vars = [u for u in var_map.keys() if u != v]
                    if other_vars:
                        terms.append(random.choice(other_vars))
                    else:
                        # Only v is available; this would be reflexive, so skip
                        terms.append(v)

            # If p is binary and ended up with ["X", "X"], skip this body atom
            if p.arity == 2 and len(terms) == 2 and terms[0] == terms[1]:
                continue

            body.append(Atom(p, terms))

        # (2) Add up to max_body_len more random body atoms, again avoiding (X, X)
        extra_len = random.randint(0, max_body_len)
        for _ in range(extra_len):
            p = random.choice(available_preds)
            if not var_map:
                break

            if p.arity == 2:
                # Pick two distinct variables if possible
                vars_list = list(var_map.keys())
                if len(vars_list) < 2:
                    # Cannot form a non‐reflexive binary atom, so skip
                    continue
                v1, v2 = random.sample(vars_list, k=2)
                terms = [v1, v2]
            else:
                # Unary or zero‐ary: sample normally
                terms = [random.choice(list(var_map.keys())) for _ in p.arg_types]

            # Double‐check reflexive case for binary
            if p.arity == 2 and terms[0] == terms[1]:
                continue

            body.append(Atom(p, terms))

        extra_rules.append(Rule(head, body))

        extra_rules = [
            r for r in extra_rules
            if r.body               # drop any Rule with body=[]
            and not _is_trivial(r)
        ]

    return extra_rules



def generate_fol_program(
    cfg: Dict[str, Any],
) -> Tuple[LogicProgram, List[Atom]]:
    d = cfg["proof_depth"]
    # pick predicate pools
    person_unary = [p for p in PREDICATE_POOL if p.arity==1 and p.arg_types[0]=="person"]
    object_unary = [p for p in PREDICATE_POOL if p.arity==1 and p.arg_types[0]=="object"]

    # 1) root predicate and constant
    if random.random()<0.5 and person_unary:
        root_pred, root_type = random.choice(person_unary), "person"
    else:
        root_pred, root_type = random.choice(object_unary), "object"
    const = random.choice(CONSTANT_POOL[root_type])
    root_fact = Atom(root_pred, [const])

    # 2) build a non‐cyclic chain_seq of length d+1
    candidates = person_unary if root_type=="person" else object_unary
    while True:
        chain_seq = [root_pred]
        for i in range(d):
            if cfg["allow_recursion"]:
                banned = {chain_seq[-1]}
                if i==d-1: banned.add(root_pred)
            else:
                banned = set(chain_seq)
            choices = [p for p in candidates if p not in banned] or candidates
            chain_seq.append(random.choice(choices))
        if chain_seq[-1]!=root_pred:
            break

    # 3) chain rules
    chain_rules = [
        Rule(Atom(chain_seq[i],["X"]), [Atom(chain_seq[i-1],["X"])])
        for i in range(1, len(chain_seq))
    ]

    # 4) sample exactly num_extra non‐trivial, non‐recursive, unique extras
    num_extra = cfg["num_rules"] - len(chain_rules)
    extra_rules, seen, attempts = [], set(), 0
    while len(extra_rules)<num_extra and attempts<RETRY_LIMIT:
        attempts += 1
        for r in _generate_extra_fol_rules(1, cfg["max_body_length"], {p.name for p in chain_seq}):
            key = repr(r)
            if (
                r.body
                and not _is_trivial(r)
                and (cfg["allow_recursion"] or r.head.predicate not in {b.predicate for b in r.body})
                and len(r.body) <= cfg["max_body_length"]
                and key not in seen
            ):
                seen.add(key)
                extra_rules.append(r)
            if len(extra_rules)==num_extra:
                break

    prog = LogicProgram(chain_rules + extra_rules)

    # 5) base facts
    other_preds = [p for p in PREDICATE_POOL if p.name not in {p.name for p in chain_seq}]
    atoms = set()
    for p in other_preds:
        if p.arity==0:
            atoms.add(Atom(p,[]))
        elif p.arity==1:
            for c in CONSTANT_POOL[p.arg_types[0]]:
                atoms.add(Atom(p,[c]))
        else:
            for c1 in CONSTANT_POOL[p.arg_types[0]]:
                for c2 in CONSTANT_POOL[p.arg_types[1]]:
                    if c1!=c2:
                        atoms.add(Atom(p,[c1,c2]))
    other_base = random.sample(list(atoms), min(cfg["num_base_facts"]-1, len(atoms)))
    base = [root_fact] + other_base

    return prog, base


# --------------------------
# Clingo Grounding & Proof Discovery
# --------------------------
# --------------------------
# Helper: Fully Ground a First‐Order Program (Robust to Type Conflicts)
# --------------------------
def _ground_rule(rule: Rule) -> List[Rule]:
    """
    Given one Rule (potentially containing variables), return a list of fully ground Rule instances
    by substituting every variable with all constants of the appropriate type.  If a single variable
    is used in positions with conflicting types, we drop that rule entirely (return an empty list).
    """
    # 1) Collect all variables and record their types.
    var_types: Dict[str, str] = {}

    def _collect_from_atom(atom: Atom):
        for idx, term in enumerate(atom.terms):
            # If this term is a “variable” (starts with uppercase letter), note its required type:
            if term and term[0].isupper():
                expected_type = atom.predicate.arg_types[idx]
                if term in var_types:
                    # If we've already seen this var, its type must match:
                    if var_types[term] != expected_type:
                        # Conflict: same var name used with different arg_types → drop rule
                        raise RuntimeError(f"Type mismatch for variable {term}")
                else:
                    var_types[term] = expected_type

    try:
        # Collect variable types from head and body:
        _collect_from_atom(rule.head)
        for b in rule.body:
            _collect_from_atom(b)
    except RuntimeError:
        # A type‐mismatch occurred; discard this rule entirely.
        return []

    # 2) If no variables, the rule is already ground:
    if not var_types:
        return [rule]

    # 3) Build the domain for each variable (cartesian product):
    var_names = list(var_types.keys())
    var_domains = [CONSTANT_POOL[var_types[v]] for v in var_names]

    ground_rules: List[Rule] = []
    for assignment in product(*var_domains):
        sub_map = {var_names[i]: assignment[i] for i in range(len(var_names))}

        def _instantiate_atom(atom: Atom) -> Atom:
            if atom.predicate.arity == 0:
                return Atom(atom.predicate, [])
            new_terms = []
            for t in atom.terms:
                if t and t[0].isupper():  # a variable
                    new_terms.append(sub_map[t])
                else:
                    new_terms.append(t)
            return Atom(atom.predicate, new_terms)

        ground_head = _instantiate_atom(rule.head)
        ground_body = [_instantiate_atom(b) for b in rule.body]
        ground_rules.append(Rule(ground_head, ground_body))

    return ground_rules


def clingo_grounding(
    program: LogicProgram, base_facts: List[Atom]
) -> Dict[str, Set[Atom]]:
    """
    Replace every variable‐containing rule with all its ground instantiations (using _ground_rule).
    Any rule that fails to ground (type conflict) is simply dropped. Then call Clingo on the
    resulting purely ground program + base_facts. Returns {"true_facts", "false_facts"}.
    """
    all_ground_rules: List[Rule] = []
    for r in program.rules:
        grounded_list = _ground_rule(r)
        # _ground_rule returns [] if there was a type conflict → effectively skips that rule
        all_ground_rules.extend(grounded_list)

    # Serialize the “clean” ground rules and base_facts:
    asp_rules = "\n".join(r.to_asp() for r in all_ground_rules)
    asp_facts = "\n".join(a.to_asp() for a in base_facts)

    # Collect all predicates that appear in these ground rules:
    preds = {r.head.predicate for r in all_ground_rules} | {
        b.predicate for r in all_ground_rules for b in r.body
    }
    shows = "\n".join(f"#show {p.name}/{p.arity}." for p in preds)

    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], asp_rules + "\n" + asp_facts + "\n" + shows)
    ctl.ground([("base", [])])

    true_strs: Set[str] = set()
    ctl.solve(on_model=lambda m: true_strs.update(str(x) for x in m.symbols(shown=True)))

    # Parse true atoms back into Atom objects
    true_atoms: Set[Atom] = set()
    for s in true_strs:
        if "(" in s:
            name, args = s.split("(", 1)
            args = args.rstrip(")").split(",")
            p = next((pp for pp in PREDICATE_POOL if pp.name == name and pp.arity == len(args)), None)
            if p:
                true_atoms.add(Atom(p, args))
        else:
            p = next((pp for pp in PREDICATE_POOL if pp.name == s and pp.arity == 0), None)
            if p:
                true_atoms.add(Atom(p, []))

    # Build the full Herbrand base for these predicates so we know which are false:
    all_atoms: Set[Atom] = set()
    for p in preds:
        if p.arity == 0:
            all_atoms.add(Atom(p, []))
        elif p.arity == 1:
            for c in CONSTANT_POOL[p.arg_types[0]]:
                all_atoms.add(Atom(p, [c]))
        else:  # arity == 2
            typ1, typ2 = p.arg_types
            for c1 in CONSTANT_POOL[typ1]:
                for c2 in CONSTANT_POOL[typ2]:
                    if c1 != c2:
                        all_atoms.add(Atom(p, [c1, c2]))

    false_atoms = all_atoms - true_atoms
    return {"true_facts": true_atoms, "false_facts": false_atoms}


def determine_how_proved(
    program: LogicProgram, base_facts: List[Atom], target: Atom
) -> Dict[str, Any]:
    known: Set[Atom] = set(base_facts)
    set_depth: Dict[str, int] = {str(a): 0 for a in base_facts}
    der_rules: Dict[str, Any] = {}
    depth = 0

    while True:
        new: Set[Atom] = set()
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

    steps: List[Dict[str, Any]] = []
    cur = str(target)
    while cur in der_rules:
        info = der_rules[cur]
        steps.append({"derived_fact": cur, **info})
        if not info["body_atoms"]:
            break
        cur = max(info["body_atoms"], key=lambda x: set_depth.get(x, 0))
    steps.reverse()
    return {"derivable": True, "steps": steps, "depth": set_depth[str(target)]}


# --------------------------
# Task Generation: Deduction
# --------------------------
def generate_multiple_deduction_tasks(
    program: LogicProgram,
    base: List[Atom],
    config: Dict[str, Any],
    n: int,
    yes_ratio: float = 0.9,
) -> List[Dict[str, Any]]:
    grounded = clingo_grounding(program, base)
    true_facts = grounded["true_facts"]
    false_facts = grounded["false_facts"]

    proofs = {
        atom: determine_how_proved(program, base, atom)
        for atom in true_facts
    }
    yes_pool = [a for a, info in proofs.items() if info["derivable"]]
    no_pool  = list(false_facts)

    num_yes = int(n * yes_ratio)
    num_no  = n - num_yes

    yes_samples = random.sample(yes_pool, min(num_yes, len(yes_pool)))
    no_samples  = random.sample(no_pool,  min(num_no,  len(no_pool)))

    tasks: List[Dict[str, Any]] = []
    for target in yes_samples + no_samples:
        label = "true" if target in yes_samples else "false"
        depth = config["proof_depth"] if label == "true" else "not applicable"
        q = target.to_asp().strip()
        c_rules = program.to_asp()
        c_facts = "\n".join(a.to_asp() for a in base)
        c = f"{c_rules}\n{c_facts}"
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


# --------------------------
# Task Generation: Abduction
# --------------------------
def generate_multiple_abduction_tasks(
    program: LogicProgram,
    base: List[Atom],
    config: Dict[str, Any],
    n: int,
    yes_ratio: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Strict “mirror‐of‐deduction” abduction:
      • The only valid “true” hypothesis is the original root_fact.
      • Any other ground atom can serve as a “false” hypothesis; it must fail to derive obs.
    """
    d = config["proof_depth"]

    # (1) Identify root_fact and separate it from the rest:
    root_fact = base[0]
    other_base = [a for a in base if a != root_fact]

    # (2) The conclusion “obs” is the final chain head applied to the same constant.
    last_rule = program.rules[d - 1]
    root_const = root_fact.terms[0]
    obs = Atom(last_rule.head.predicate, [root_const])

    # (3) Build lists of “true_candidates” and “false_candidates.”
    #     → True candidate = exactly {root_fact}
    true_candidates = [root_fact]

    #     → False candidates = all ground atoms whose predicate is in chain_preds[:-1],
    #       but we exclude root_fact itself.  (And ensure they never appear in base.)
    chain_preds: List[Predicate] = []
    for idx in range(d):
        rule = program.rules[idx]
        if idx == 0:
            chain_preds.append(rule.body[0].predicate)
        chain_preds.append(rule.head.predicate)

    # We will propose any ground atom of “candidate_preds” that is NOT root_fact as a false hypothesis:
    candidate_preds = chain_preds[:-1]  # root_pred and intermediates

    false_candidates: List[Atom] = []
    for p in candidate_preds:
        # Ground this predicate over all valid constants of its argument types:
        if p.arity == 1:
            typ = p.arg_types[0]
            for c in CONSTANT_POOL[typ]:
                atom = Atom(p, [c])
                if atom != root_fact and atom not in base:
                    false_candidates.append(atom)
        elif p.arity == 2:
            typ1, typ2 = p.arg_types
            for c1 in CONSTANT_POOL.get(typ1, []):
                for c2 in CONSTANT_POOL.get(typ2, []):
                    if c1 != c2:
                        atom = Atom(p, [c1, c2])
                        if atom != root_fact and atom not in base:
                            false_candidates.append(atom)

    # (4) Now test which of those actually yield “true” vs. “false.”
    yes_pool: List[Atom] = []
    no_pool: List[Atom] = []

    # --- Check root_fact as a “true” hypothesis: ---
    # Without root_fact: can we derive obs?
    grounded_core = clingo_grounding(program, other_base)
    if obs not in grounded_core["true_facts"]:
        # If removing root_fact breaks the chain, check that re-adding root_fact restores it:
        grounded_with_root = clingo_grounding(program, other_base + [root_fact])
        if obs in grounded_with_root["true_facts"]:
            yes_pool.append(root_fact)
        else:
            # In the extremely rare case that re-adding root_fact STILL doesn't derive obs,
            # we simply put it into the no_pool.  (This shouldn’t happen if the chain is built correctly.)
            no_pool.append(root_fact)

    # --- Check each false_candidate: they should all end up in no_pool, 
    #     because none of them is the actual root. ---
    for h in false_candidates:
        # If obs is already derivable without h, skip it (it’s a “trivially false” candidate).
        # We want only those that don’t break the chain in the first place:
        grounded_core = clingo_grounding(program, other_base)
        if obs in grounded_core["true_facts"]:
            # If conclusion is already derivable without root_fact, skip h entirely.
            continue

        # Now test adding h:
        grounded_with_h = clingo_grounding(program, other_base + [h])
        if obs not in grounded_with_h["true_facts"]:
            no_pool.append(h)

    # (5) Sample exactly n tasks from yes_pool/ no_pool according to yes_ratio:
    num_yes = min(int(n * yes_ratio), len(yes_pool))
    num_no  = n - num_yes

    yes_samples = random.sample(yes_pool, num_yes) if yes_pool else []
    no_samples  = random.sample(no_pool,  num_no)  if no_pool  else []

    tasks: List[Dict[str, Any]] = []
    for hidden in yes_samples + no_samples:
        label = "true" if hidden in yes_samples else "false"

        # The context shown to the LLM *still must include obs*, plus all other_base:
        context_facts = [obs] + other_base

        q = hidden.to_asp().strip()
        c_rules = program.to_asp()
        c_facts = "\n".join(a.to_asp() for a in context_facts)
        c = f"{c_rules}\n{c_facts}"

        nl = generate_abduction_prompt(program, context_facts, hidden, obs)
        depth_val = config["proof_depth"] if label == "true" else "not applicable"
        tasks.append({
            "q": q,
            "c": c,
            "natural language": nl,
            "t": label,
            "metadata": {
                **config,
                "depth": depth_val,
                "reasoning_type": "abduction",
            },
        })

    return tasks


# --------------------------
# Prompt Generation
# --------------------------
def generate_deduction_prompt(
    program: LogicProgram, base: List[Atom], target: Atom
) -> str:
    rules_nl = [r.to_nl() for r in program.rules]
    facts_nl = [f"{a.to_nl()}." for a in base]
    return (
        "You are given the following information:\n"
        + "\n".join(rules_nl)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts_nl)
        + f"\n\nQUESTION:\nIs {target.to_nl()} true?"
        + "\nAnswer exactly true or false."
    )


def generate_abduction_prompt(
    program: LogicProgram, ctx: List[Atom], hypothesis: Atom, obs: Atom
) -> str:
    rules_nl = [r.to_nl() for r in program.rules]
    facts_nl = [f"{a.to_nl()}." for a in ctx]
    return (
        "You are given the following information:\n"
        + "\n".join(rules_nl)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts_nl)
        + f"\n\nQUESTION:\nCould {hypothesis.to_nl()} explain {obs.to_nl()}?"
        + "\nAnswer exactly true or false."
    )


# --------------------------
# Main Driver: Generate & Save Benchmark
# --------------------------
if __name__ == "__main__":
    NUM_RULES      = [5, 15, 25, 35]
    NUM_BASE_FACTS = [4, 8, 12, 16, 20]
    PROOF_DEPTHS   = [1, 5, 10, 20, 30]
    RECURSION_OPTIONS = [True, False]
    TASKS_PER_GROUP = 5
    GROUPS_PER_CFG  = 1

    all_tasks: List[Dict[str, Any]] = []
    for nr, nb, pd, ro in product(NUM_RULES, NUM_BASE_FACTS, PROOF_DEPTHS, RECURSION_OPTIONS):
        if pd > nr:
            continue
        cfg = {
            "num_rules":        nr,
            "max_body_length":  3,
            "allow_recursion":  ro,
            "branching_factor": 2,
            "proof_depth":      pd,
            "num_base_facts":   nb,
        }
        for _ in range(GROUPS_PER_CFG):
            prog, base = generate_fol_program(cfg)
            ded_tasks = generate_multiple_deduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
            abd_tasks = generate_multiple_abduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
            all_tasks.extend(ded_tasks)
            all_tasks.extend(abd_tasks)

    with open("first_order_benchmark.json", "w") as f:
        json.dump(all_tasks, f, indent=2)

    print(f"Generated {len(all_tasks)} total tasks in first_order_benchmark.json")
