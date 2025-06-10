import logging
import random
import json
from typing import List, Tuple, Optional, Dict, Set, Any
import clingo
from itertools import product

# Setup
RETRY_LIMIT: int = 200
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
VARIABLE_POOL = ["X", "Y", "Z", "U", "V", "W", "A", "B", "C", "D", "E", "F"]

def _var_name(typ: str, var: str) -> str:
    return f"{typ} {var}"

# Core Classes
class Predicate:
    def __init__(self, name: str, arity: int, arg_types: Optional[List[str]] = None, nl_template: Optional[str] = None):
        self.name: str = name
        self.arity: int = arity
        self.arg_types: List[str] = arg_types if arg_types is not None else ["any"] * arity
        self.nl_template: str = nl_template if nl_template is not None else self._default_template()

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
        assert len(terms) == predicate.arity
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
                typ = self.predicate.arg_types[i]
                words.append(_var_name(typ, t))
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

    def get_variables(self) -> Dict[str, Set[str]]:
        """Return sets of variables in head and body"""
        head_vars = {t for t in self.head.terms if t and t[0].isupper()}
        body_vars = {t for atom in self.body for t in atom.terms if t and t[0].isupper()}
        return {"head": head_vars, "body": body_vars}

    def to_nl(self) -> str:
        """Generate natural language with proper quantification"""
        if self.is_fact():
            return f"{self.head.to_nl()}."
        
        var_info = self.get_variables()
        head_vars = sorted(var_info["head"])
        existential_vars = sorted(var_info["body"] - var_info["head"])
        
        # Build quantification
        parts = []
        if head_vars:
            if len(head_vars) == 1:
                parts.append(f"For all {head_vars[0]}")
            else:
                parts.append(f"For all {', '.join(head_vars)}")
        
        # Build condition
        body_conditions = [atom.to_nl() for atom in self.body]
        if existential_vars:
            if len(existential_vars) == 1:
                exists_part = f"there exists {existential_vars[0]} such that"
            else:
                exists_part = f"there exist {', '.join(existential_vars)} such that"
            condition = f"if {exists_part} {' and '.join(body_conditions)}"
        else:
            condition = f"if {' and '.join(body_conditions)}"
        
        conclusion = f"then {self.head.to_nl()}"
        
        if parts:
            return f"{', '.join(parts)}, {condition}, {conclusion}."
        else:
            return f"If {' and '.join(body_conditions)}, {conclusion}."

class LogicProgram:
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = rules if rules is not None else []

    def to_asp(self) -> str:
        return "\n".join(r.to_asp() for r in self.rules)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

# Predicate pools
PREDICATE_POOL: List[Predicate] = [
    Predicate("sunny", 0, [], "it is sunny"),
    Predicate("cold", 0, [], "it is cold"),
    Predicate("hot", 0, [], "it is hot"),
    Predicate("wet", 1, ["object"], "{0} is wet"),
    Predicate("big", 1, ["object"], "{0} is big"),
    Predicate("small", 1, ["object"], "{0} is small"),
    Predicate("clean", 1, ["object"], "{0} is clean"),
    Predicate("red", 1, ["object"], "{0} is red"),
    Predicate("heavy", 1, ["object"], "{0} is heavy"),
    Predicate("light", 1, ["object"], "{0} is light"),
    Predicate("soft", 1, ["object"], "{0} is soft"),
    Predicate("hard", 1, ["object"], "{0} is hard"),
    Predicate("smooth", 1, ["object"], "{0} is smooth"),
    Predicate("rough", 1, ["object"], "{0} is rough"),
    Predicate("new", 1, ["object"], "{0} is new"),
    Predicate("old", 1, ["object"], "{0} is old"),
    Predicate("dirty", 1, ["object"], "{0} is dirty"),
    Predicate("sad", 1, ["person"], "{0} is sad"),
    Predicate("tall", 1, ["person"], "{0} is tall"),
    Predicate("happy", 1, ["person"], "{0} is happy"),
    Predicate("hungry", 1, ["person"], "{0} is hungry"),
    Predicate("calm", 1, ["person"], "{0} is calm"),
    Predicate("excited", 1, ["person"], "{0} is excited"),
    Predicate("angry", 1, ["person"], "{0} is angry"),
    Predicate("brave", 1, ["person"], "{0} is brave"),
    Predicate("clever", 1, ["person"], "{0} is clever"),
    Predicate("curious", 1, ["person"], "{0} is curious"),
    Predicate("tired", 1, ["person"], "{0} is tired"),
    Predicate("friendly", 1, ["person"], "{0} is friendly"),
    Predicate("strong", 1, ["person"], "{0} is strong"),
    Predicate("weak", 1, ["person"], "{0} is weak"),
    Predicate("bored", 1, ["person"], "{0} is bored"),
    Predicate("busy", 1, ["person"], "{0} is busy"),
    Predicate("funny", 1, ["person"], "{0} is funny"),
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

def create_existential_rule(available_preds: List[Predicate], max_body_len: int) -> Optional[Rule]:
    """Create a rule with meaningful existential variables and consistent types"""
    if len(available_preds) < 3:
        return None
    
    head_pred = random.choice(available_preds)
    
    # Create head and track variable types
    var_types = {}  # Maps variable -> type (person/object)
    
    if head_pred.arity == 0:
        head = Atom(head_pred, [])
        head_vars = set()
    elif head_pred.arity == 1:
        head_var = random.choice(VARIABLE_POOL[:3])  # Use X, Y, Z for head
        var_types[head_var] = head_pred.arg_types[0]
        head = Atom(head_pred, [head_var])
        head_vars = {head_var}
    else:  # arity == 2
        head_var1, head_var2 = random.sample(VARIABLE_POOL[:3], 2)
        var_types[head_var1] = head_pred.arg_types[0]
        var_types[head_var2] = head_pred.arg_types[1]
        head = Atom(head_pred, [head_var1, head_var2])
        head_vars = {head_var1, head_var2}
    
    if not head_vars:
        return None
    
    body = []
    
    # Strategy 1: Ensure ALL head variables appear in body with consistent types
    for head_var in head_vars:
        var_type = var_types[head_var]
        # Find predicates that can use this variable type
        compatible_preds = [p for p in available_preds if p.arity == 1 and p.arg_types[0] == var_type]
        if compatible_preds:
            condition_pred = random.choice(compatible_preds)
            body.append(Atom(condition_pred, [head_var]))
    
    # Strategy 2: Add meaningful existential variables with proper types
    if len(body) < max_body_len and random.random() < 0.6:
        # Pick a head variable to connect through existential
        head_var = random.choice(list(head_vars))
        head_var_type = var_types[head_var]
        
        # Create existential bridge variable
        bridge_var = random.choice(VARIABLE_POOL[3:8])
        
        # Find binary predicates that can connect head_var to bridge_var
        compatible_binary_preds = []
        for p in available_preds:
            if p.arity == 2:
                # Check if head_var can be first argument
                if p.arg_types[0] == head_var_type:
                    bridge_type = p.arg_types[1]
                    compatible_binary_preds.append((p, 0, bridge_type))  # head_var at position 0
                # Check if head_var can be second argument
                elif p.arg_types[1] == head_var_type:
                    bridge_type = p.arg_types[0]
                    compatible_binary_preds.append((p, 1, bridge_type))  # head_var at position 1
        
        if compatible_binary_preds:
            pred, head_var_pos, bridge_type = random.choice(compatible_binary_preds)
            var_types[bridge_var] = bridge_type
            
            # Create bridge atom with correct argument order
            if head_var_pos == 0:
                bridge_atom = Atom(pred, [head_var, bridge_var])
            else:
                bridge_atom = Atom(pred, [bridge_var, head_var])
            
            body.append(bridge_atom)
            
            # Add condition on bridge variable
            bridge_condition_preds = [p for p in available_preds if p.arity == 1 and p.arg_types[0] == bridge_type]
            if bridge_condition_preds and len(body) < max_body_len:
                bridge_condition_pred = random.choice(bridge_condition_preds)
                body.append(Atom(bridge_condition_pred, [bridge_var]))
    
    # Strategy 3: Add one more existential condition (optional)
    if len(body) < max_body_len and random.random() < 0.3:
        existential_var = random.choice(VARIABLE_POOL[6:])
        if existential_var not in var_types:
            # Pick a random type for this existential variable
            existential_type = random.choice(["person", "object"])
            var_types[existential_var] = existential_type
            
            condition_preds = [p for p in available_preds if p.arity == 1 and p.arg_types[0] == existential_type]
            if condition_preds:
                condition_pred = random.choice(condition_preds)
                body.append(Atom(condition_pred, [existential_var]))
    
    # VALIDATION: Check that all head variables appear in body
    body_vars = {t for atom in body for t in atom.terms if t and t[0].isupper()}
    if not head_vars.issubset(body_vars):
        return None  # Invalid rule, reject it
    
    # VALIDATION: Check type consistency
    for atom in body:
        for i, term in enumerate(atom.terms):
            if term in var_types:
                expected_type = atom.predicate.arg_types[i]
                actual_type = var_types[term]
                if expected_type != actual_type:
                    return None  # Type mismatch, reject rule
    
    if not body:
        return None
    
    return Rule(head, body)

def create_universal_rule(available_preds: List[Predicate]) -> Optional[Rule]:
    """Create a simple universal rule with consistent types"""
    if len(available_preds) < 2:
        return None
    
    head_pred = random.choice(available_preds)
    
    if head_pred.arity == 0:
        return None
    
    # Track variable types for consistency
    var_types = {}
    
    if head_pred.arity == 1:
        var = "X"
        var_types[var] = head_pred.arg_types[0]
        head = Atom(head_pred, [var])
        
        # Add 1-2 body conditions using the same variable and type
        body = []
        compatible_preds = [p for p in available_preds if p.arity == 1 and p.arg_types[0] == var_types[var] and p != head_pred]
        if compatible_preds:
            body.append(Atom(random.choice(compatible_preds), [var]))
        
        if random.random() < 0.4 and compatible_preds:
            body.append(Atom(random.choice(compatible_preds), [var]))
        
        return Rule(head, body) if body else None
    
    else:  # arity == 2
        var1, var2 = "X", "Y"
        var_types[var1] = head_pred.arg_types[0]
        var_types[var2] = head_pred.arg_types[1]
        head = Atom(head_pred, [var1, var2])
        
        body = []
        
        # Add conditions on both variables with correct types
        for var in [var1, var2]:
            var_type = var_types[var]
            compatible_preds = [p for p in available_preds if p.arity == 1 and p.arg_types[0] == var_type]
            if compatible_preds:
                body.append(Atom(random.choice(compatible_preds), [var]))
        
        # Optionally add a relationship between the two variables
        if random.random() < 0.3:
            relationship_preds = [p for p in available_preds 
                                if p.arity == 2 
                                and p.arg_types[0] == var_types[var1] 
                                and p.arg_types[1] == var_types[var2]
                                and p != head_pred]
            if relationship_preds:
                body.append(Atom(random.choice(relationship_preds), [var1, var2]))
        
        return Rule(head, body) if body else None

def generate_enhanced_fol_program(cfg: Dict[str, Any]) -> Tuple[LogicProgram, List[Atom], List[Predicate]]:
    """Generate FOL program with mix of universal and existential quantification"""
    d = cfg["proof_depth"]
    
    # Choose root type and create chain
    person_unary = [p for p in PREDICATE_POOL if p.arity == 1 and p.arg_types[0] == "person"]
    object_unary = [p for p in PREDICATE_POOL if p.arity == 1 and p.arg_types[0] == "object"]
    
    if random.random() < 0.5 and person_unary:
        root_pred, root_type = random.choice(person_unary), "person"
        candidates = person_unary
    else:
        root_pred, root_type = random.choice(object_unary), "object"
        candidates = object_unary
    
    const = random.choice(CONSTANT_POOL[root_type])
    root_fact = Atom(root_pred, [const])
    
    # Build chain sequence
    while True:
        chain_seq = [root_pred]
        for i in range(d):
            if cfg["allow_recursion"]:
                banned = {chain_seq[-1]}
                if i == d-1: banned.add(root_pred)
            else:
                banned = set(chain_seq)
            choices = [p for p in candidates if p not in banned] or candidates
            chain_seq.append(random.choice(choices))
        if chain_seq[-1] != root_pred:
            break
    
    # Create chain rules (these will be universal)
    chain_rules = [
        Rule(Atom(chain_seq[i], ["X"]), [Atom(chain_seq[i-1], ["X"])])
        for i in range(1, len(chain_seq))
    ]
    
    # Generate extra rules with mix of universal and existential
    num_extra = cfg["num_rules"] - len(chain_rules)
    extra_rules = []
    available_preds = [p for p in PREDICATE_POOL if p.name not in {p.name for p in chain_seq}]
    
    # Target: ~40% existential rules, 60% universal rules
    num_existential = int(num_extra * 0.4)
    num_universal = num_extra - num_existential
    
    attempts = 0
    while len(extra_rules) < num_extra and attempts < RETRY_LIMIT:
        attempts += 1
        
        if len([r for r in extra_rules if r.get_variables()["body"] - r.get_variables()["head"]]) < num_existential:
            # Create existential rule
            rule = create_existential_rule(available_preds, cfg["max_body_length"])
        else:
            # Create universal rule
            rule = create_universal_rule(available_preds)
        
        if rule and rule.body:  # Ensure non-trivial rule
            extra_rules.append(rule)
    
    prog = LogicProgram(chain_rules + extra_rules)
    
    # Generate base facts
    other_preds = [p for p in PREDICATE_POOL if p.name not in {p.name for p in chain_seq}]
    atoms = set()
    for p in other_preds:
        if p.arity == 0:
            atoms.add(Atom(p, []))
        elif p.arity == 1:
            for c in CONSTANT_POOL[p.arg_types[0]]:
                atoms.add(Atom(p, [c]))
        else:
            for c1 in CONSTANT_POOL[p.arg_types[0]]:
                for c2 in CONSTANT_POOL[p.arg_types[1]]:
                    if c1 != c2:
                        atoms.add(Atom(p, [c1, c2]))
    
    other_base = random.sample(list(atoms), min(cfg["num_base_facts"]-1, len(atoms)))
    base = [root_fact] + other_base
    
    return prog, base, chain_seq

# Clingo solving (simplified version)
def solve_program(program: LogicProgram, base_facts: List[Atom]) -> Set[Atom]:
    """Simple forward chaining solver"""
    known = set(base_facts)
    changed = True
    
    while changed:
        changed = False
        for rule in program.rules:
            if rule.is_fact():
                continue
            
            # Try all possible variable assignments
            var_info = rule.get_variables()
            all_vars = var_info["head"] | var_info["body"]
            
            if not all_vars:
                continue
            
            # Simple grounding for demonstration
            for person in CONSTANT_POOL["person"]:
                for obj in CONSTANT_POOL["object"]:
                    # Try assignment
                    assignment = {}
                    var_list = list(all_vars)
                    for i, var in enumerate(var_list):
                        if i < len(CONSTANT_POOL["person"]):
                            assignment[var] = person if i == 0 else obj
                        else:
                            assignment[var] = obj
                    
                    # Check if body is satisfied
                    body_satisfied = True
                    for body_atom in rule.body:
                        ground_terms = []
                        for term in body_atom.terms:
                            if term in assignment:
                                ground_terms.append(assignment[term])
                            else:
                                ground_terms.append(term)
                        ground_atom = Atom(body_atom.predicate, ground_terms)
                        if ground_atom not in known:
                            body_satisfied = False
                            break
                    
                    if body_satisfied:
                        # Add head
                        head_terms = []
                        for term in rule.head.terms:
                            if term in assignment:
                                head_terms.append(assignment[term])
                            else:
                                head_terms.append(term)
                        head_atom = Atom(rule.head.predicate, head_terms)
                        if head_atom not in known:
                            known.add(head_atom)
                            changed = True
    
    return known

def generate_deduction_tasks(program: LogicProgram, base: List[Atom], chain_seq: List[Predicate], 
                           config: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Generate deduction tasks"""
    d = config["proof_depth"]
    const = base[0].terms[0]
    
    # Positive target
    head_pred = chain_seq[d]
    positive = Atom(head_pred, [const])
    
    # Negative samples
    all_atoms = []
    for p in PREDICATE_POOL:
        if p.arity == 1:
            for c in CONSTANT_POOL[p.arg_types[0]]:
                atom = Atom(p, [c])
                if atom not in base and atom != positive:
                    all_atoms.append(atom)
    
    negatives = random.sample(all_atoms, min(n-1, len(all_atoms)))
    
    tasks = []
    for atom in [positive] + negatives:
        label = "true" if atom == positive else "false"
        
        q = atom.to_asp().strip()
        c_rules = program.to_asp()
        c_facts = "\n".join(a.to_asp() for a in base)
        
        rules_nl = [r.to_nl() for r in program.rules]
        facts_nl = [f"{a.to_nl()}." for a in base]
        nl = (
            "You are given the following information:\n" +
            "\n".join(rules_nl) +
            "\n\nAnd the following facts:\n" +
            "\n".join(facts_nl) +
            f"\n\nQUESTION:\nIs {atom.to_nl()} true?\nAnswer exactly true or false."
        )
        
        tasks.append({
            "q": q,
            "c": c_rules + "\n" + c_facts,
            "natural language": nl,
            "t": label,
            "metadata": {
                **config,
                "depth": d if label == "true" else "not applicable",
                "reasoning_type": "deduction",
            },
        })
    
    return tasks

def generate_abduction_tasks(program: LogicProgram, base: List[Atom], config: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Generate abduction tasks"""
    d = config["proof_depth"]
    
    root_fact = base[0]
    other_base = [a for a in base if a != root_fact]
    
    # The observation is the end of the chain
    last_rule = program.rules[d - 1]
    root_const = root_fact.terms[0]
    obs = Atom(last_rule.head.predicate, [root_const])
    
    # Context facts for the task (including observation)
    context_facts = [obs] + other_base
    
    # Generate hypotheses
    true_candidates = [root_fact]
    false_candidates = []
    
    # Add some false candidates
    for p in PREDICATE_POOL:
        if p.arity == 1:
            for c in CONSTANT_POOL[p.arg_types[0]]:
                atom = Atom(p, [c])
                if atom != root_fact and atom not in base:
                    false_candidates.append(atom)
    
    false_samples = random.sample(false_candidates, min(n-1, len(false_candidates)))
    
    tasks = []
    for hypothesis in true_candidates + false_samples:
        label = "true" if hypothesis in true_candidates else "false"
        
        q = hypothesis.to_asp().strip()
        c_rules = program.to_asp()
        c_facts = "\n".join(a.to_asp() for a in context_facts)
        
        rules_nl = [r.to_nl() for r in program.rules]
        facts_nl = [f"{a.to_nl()}." for a in context_facts]
        nl = (
            "You are given the following information:\n" +
            "\n".join(rules_nl) +
            "\n\nAnd the following facts:\n" +
            "\n".join(facts_nl) +
            f"\n\nQUESTION:\nCould {hypothesis.to_nl()} explain {obs.to_nl()}?\nAnswer exactly true or false."
        )
        
        tasks.append({
            "q": q,
            "c": c_rules + "\n" + c_facts,
            "natural language": nl,
            "t": label,
            "metadata": {
                **config,
                "depth": config["proof_depth"] if label == "true" else "not applicable",
                "reasoning_type": "abduction",
            },
        })
    
    return tasks

# Main generation
def generate_benchmark():
    """Generate the complete benchmark"""
    NUM_RULES = [5, 10, 15, 20, 25]
    NUM_BASE_FACTS = [4, 8, 12, 16, 20]
    PROOF_DEPTHS = [1, 5, 10, 20]
    RECURSION_OPTIONS = [True, False]
    TASKS_PER_GROUP = 4
    GROUPS_PER_CFG = 1

    all_tasks = []
    task_id = 0
    
    for nr, nb, pd, ro in product(NUM_RULES, NUM_BASE_FACTS, PROOF_DEPTHS, RECURSION_OPTIONS):
        if pd > nr:
            continue
        
        cfg = {
            "num_rules": nr,
            "max_body_length": 3,
            "allow_recursion": ro,
            "branching_factor": 2,
            "proof_depth": pd,
            "num_base_facts": nb,
        }
        
        for _ in range(GROUPS_PER_CFG):
            try:
                prog, base, chain_seq = generate_enhanced_fol_program(cfg)
                
                ded_tasks = generate_deduction_tasks(prog, base, chain_seq, cfg, TASKS_PER_GROUP)
                abd_tasks = generate_abduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
                
                # Add IDs
                for task in ded_tasks + abd_tasks:
                    task["id"] = task_id
                    task_id += 1
                
                all_tasks.extend(ded_tasks)
                all_tasks.extend(abd_tasks)
                
            except Exception as e:
                logger.warning(f"Failed to generate for config {cfg}: {e}")
                continue
    
    return all_tasks

if __name__ == "__main__":
    print("Generating new benchmark with proper existential quantification...")
    
    tasks = generate_benchmark()
    
    # Count existential rules
    existential_count = 0
    total_rules = 0
    
    for task in tasks:
        c_text = task.get('c', '')
        for line in c_text.split('\n'):
            if ':-' in line:
                total_rules += 1
                # Check if natural language has "there exists"
                nl = task.get('natural language', '')
                if 'there exist' in nl:
                    existential_count += 1
                    break
    
    with open("new_first_order_benchmark_with_existentials.json", "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Generated {len(tasks)} tasks")
    print(f"Estimated existential rules: {existential_count}/{total_rules}")
    print(f"Saved to: new_first_order_benchmark_with_existentials.json")