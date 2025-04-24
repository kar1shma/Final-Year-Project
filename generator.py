from typing import List, Optional, Tuple
import uuid
import random
import networkx as nx
import matplotlib.pyplot as plt


# fully parametric, compositional system, capable of sampling any logic program structure within a vast and expandable universe

# Term: Variable (e.g X, Y) or a constant (e.g. socrates, apple)
class Term:
    def __init__(self, name: str, is_variable: bool):
        self.name = name
        self.is_variable = is_variable  # True = variable, False = constant

    def __repr__(self):
        return self.name


# Predicate: Symbol like wet/1, owns/2, is_red/1, with name, arity, argument types, and metadata for POS tagging (i.e. noun, adjective, etc.)
class Predicate:
    def __init__(self, name: str, arity: int, arg_types: Optional[List[str]] = None):
        self.name = name
        self.arity = arity
        self.arg_types = arg_types or ["any"] * arity

    def __repr__(self):
        return f"{self.name}/{self.arity}"

# Example predicates:    
# Predicate("owns", arity=2, arg_types=["person", "object"])
# Predicate("wet", arity=1, arg_types=["object"])


# Atom: Predicate applied to terms, e.g. wet(X), owns(X, Y), is_red(apple)
class Atom:
    def __init__(self, predicate: Predicate, terms: List[Term]):
        assert len(terms) == predicate.arity, "Arity mismatch"
        self.predicate = predicate
        self.terms = terms

    def __repr__(self):
        args = ", ".join(map(str, self.terms))
        return f"{self.predicate.name}({args})"


# Rule: Logical rule of the form head :- body where head is an atom and body is a list of atoms (body can be empty -> fact)
class Rule:
    def __init__(self, head: Atom, body: Optional[List[Atom]] = None):
        self.head = head
        self.body = body or []  # empty => fact

    def is_fact(self) -> bool:
        return len(self.body) == 0

    def __repr__(self):
        if self.is_fact():
            return f"{self.head}."
        body_str = ", ".join(map(str, self.body))
        return f"{self.head} :- {body_str}."


# Collection of rules represented as a list of Rule objects, potentially with metadata or annotations (like loop presence).
class LogicProgram:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def __repr__(self):
        return "\n".join(map(str, self.rules))


# Define a pool of predicates with their arities and argument types
PREDICATE_POOL = [
    Predicate("sunny", 0),
    Predicate("wet", 1, arg_types=["object"]),
    Predicate("red", 1, arg_types=["object"]),
    Predicate("big", 1, arg_types=["object"]),
    Predicate("small", 1, arg_types=["object"]),
    Predicate("person", 1, arg_types=["person"]),
    Predicate("happy", 1, arg_types=["person"]),
    Predicate("sad", 1, arg_types=["person"]),
    Predicate("tall", 1, arg_types=["person"]),
    Predicate("owns", 2, arg_types=["person", "object"]),
    Predicate("likes", 2, arg_types=["person", "entity"]),
    Predicate("taller_than", 2, arg_types=["person", "person"]),
    Predicate("dislikes", 2, arg_types=["person", "object"]),
]

# Define a pool of constants for each type
CONSTANT_POOL = {
    "person": ["socrates", "plato", "alice", "bob", "charlie", "dave", "eve"],
    "object": ["apple", "book", "ball", "car", "house", "computer", "phone"],
    "entity": ["socrates", "plato", "alice", "bob", "apple", "book", "ball", "car", "house", "computer", "phone"],
}


GENERATOR_CONFIG = {
    "num_rules": 50,
    "max_body_length": 2,
    "allow_recursion": True,
    "allow_loops": True,
    "branching_factor": 2  # max num of rules with same head predicate
}


# Pool of variable names
VARIABLE_NAMES = ["X", "Y", "Z", "W", "A", "B", "C"]

def generate_logic_program(predicate_pool, constant_pool, config) -> LogicProgram:
    rules = []
    predicate_heads_used = dict() # predicate name -> num times used as head

    for _ in range(config["num_rules"]):
        # Decide whether to reuse a previous head (for branching)
        reuse = random.random() < 0.5
        if reuse and predicate_heads_used:
            # Pick a predicate that hasn't hit the branching limit
            reusable_predicates = [p for p in predicate_heads_used if predicate_heads_used[p] < config["branching_factor"]]
            if reusable_predicates:
                head_predicate_name = random.choice(reusable_predicates)  # pick a predicate that has been used before
                head_predicate = next(p for p in predicate_pool if p.name == head_predicate_name)  # find the predicate in the predicate pool
            else:
                head_predicate = random.choice(predicate_pool)
        else:
            head_predicate = random.choice(predicate_pool)

        rule = generate_rule(predicate_pool, config, head_predicate)
        rules.append(rule)

        # Track how many times we've used this predicate as a head
        predicate_heads_used[head_predicate.name] = predicate_heads_used.get(head_predicate.name, 0) + 1

    return LogicProgram(rules)


# def generate_rule(predicate_pool: List[Predicate], config: dict, head_pred: Predicate) -> Rule:
#     head_terms, var_pool = generate_terms_for_predicate(head_pred)
#     head_atom = Atom(head_pred, head_terms)

#     body_len = random.randint(0, config["max_body_length"])
#     body_atoms = []

#     for i in range(body_len):
#         if config["allow_recursion"] and random.random() < 0.3:
#             pred = head_pred  # recursion
#         else:
#             pred = random.choice(predicate_pool)

#         terms = generate_terms_for_predicate(pred, available_vars=var_pool)[0]
#         body_atoms.append(Atom(pred, terms))

#     return Rule(head=head_atom, body=body_atoms)


def generate_rule(predicate_pool: List[Predicate],
                  config: dict,
                  head_pred: Predicate) -> Rule:
    # 1. Build the head
    head_terms, var_pool = generate_terms_for_predicate(head_pred)
    head_atom = Atom(head_pred, head_terms)

    # 2. Sample the raw body
    body_len = random.randint(0, config["max_body_length"])
    body_atoms: List[Atom] = []
    for _ in range(body_len):
        pred = head_pred if (config["allow_recursion"] and random.random() < 0.3) else random.choice(predicate_pool)
        terms, var_pool = generate_terms_for_predicate(pred, available_vars=var_pool)
        body_atoms.append(Atom(pred, terms))

    # 3. Enforce range-restriction: every head variable must appear elsewhere in the body.
    head_var_names = {t.name for t in head_terms}
    body_var_names = {v.name for atom in body_atoms for v in atom.terms}
    missing = head_var_names - body_var_names

    for mv in missing:
        # find its position and required type
        idx = next(i for i,t in enumerate(head_terms) if t.name == mv)
        var_type = head_pred.arg_types[idx]

        # pick a *different* unary predicate of the same type
        candidates = [
            p for p in predicate_pool
            if p.arity == 1 
            and p.arg_types[0] == var_type
            and p.name != head_pred.name      # exclude the head predicate itself
        ]
        if not candidates:
            # if no other unary predicate, *skip* constraining mv
            continue

        constr_pred = random.choice(candidates)
        # reuse the Term from var_pool
        term = next(t for t in var_pool if t.name == mv)
        body_atoms.append(Atom(constr_pred, [term]))

    return Rule(head=head_atom, body=body_atoms)


def generate_terms_for_predicate(predicate: Predicate, available_vars=None) -> Tuple[List[Term], List[Term]]:
    """
    Returns a list of Terms that satisfy the predicate's arity and types.
    Also returns the pool of all variables used so far.
    """
    terms = []
    var_pool = available_vars[:] if available_vars else []
    used_names = {v.name for v in var_pool}

    for i in range(predicate.arity):
        # pick a variable not already used (for clarity)
        new_var_name = next(v for v in VARIABLE_NAMES if v not in used_names)
        new_var = Term(new_var_name, is_variable=True)
        terms.append(new_var)
        var_pool.append(new_var)
        used_names.add(new_var_name)

    return terms, var_pool


def instantiate_rule_with_constants(rule: Rule, constant_pool: dict) -> Rule:
    var_to_const = {}

    def get_constant_for_type(arg_type: str) -> str:
        # If predicate accepts anything, pick from a unified pool
        if arg_type == "any":
            all_constants = sum(constant_pool.values(), [])
            return random.choice(all_constants)
        return random.choice(constant_pool.get(arg_type, []))

    def instantiate_atom(atom: Atom) -> Atom:
        new_terms = []
        for i, term in enumerate(atom.terms):
            if term.name not in var_to_const:
                expected_type = atom.predicate.arg_types[i]
                chosen_const = get_constant_for_type(expected_type)
                var_to_const[term.name] = Term(name=chosen_const, is_variable=False)
            new_terms.append(var_to_const[term.name])
        return Atom(atom.predicate, new_terms)

    grounded_head = instantiate_atom(rule.head)
    grounded_body = [instantiate_atom(atom) for atom in rule.body]

    return Rule(head=grounded_head, body=grounded_body)


def build_grounded_world_graph(logic_program: LogicProgram):
    G = nx.DiGraph()

    for rule in logic_program.rules:
        atoms = [rule.head] + rule.body  # Include head + all body atoms
        for atom in atoms:
            pred = atom.predicate
            if pred.arity == 2:
                # Binary predicate: create edge with label
                src = atom.terms[0].name
                tgt = atom.terms[1].name
                G.add_edge(src, tgt, label=pred.name)
            elif pred.arity == 1:
                # Unary predicate: add self-loop or annotate
                node = atom.terms[0].name
                G.add_node(node)
                G.add_edge(node, node, label=pred.name)
            elif pred.arity == 0:
                # Nullary: global fact, skip
                pass

    return G

def draw_grounded_world_graph(G):
    plt.figure(figsize=(20, 8))
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=8, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray")
    plt.title("Grounded World Graph")
    plt.show()
    plt.savefig("grounded_world_graph.png")


def build_dependency_graph(logic_program: LogicProgram):
    G = nx.DiGraph()
    for rule in logic_program.rules:
        head_name = rule.head.predicate.name
        for atom in rule.body:
            body_name = atom.predicate.name
            G.add_edge(head_name, body_name)
    return G

def draw_dependency_graph(G):
    plt.figure(figsize=(15, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=700, font_size=10, font_weight="bold", arrows=True)
    plt.title("Predicate Dependency Graph")
    plt.show()
    plt.savefig("dependency_graph.png")



def main():
    # Generate logic program
    logic_program = generate_logic_program(PREDICATE_POOL, CONSTANT_POOL, GENERATOR_CONFIG)

    # Print the generated logic program
    print(logic_program)

    # Generate dependency graph for abstract (non-grounded) logic program
    dependency_graph = build_dependency_graph(logic_program)
    draw_dependency_graph(dependency_graph)

    # Ground the logic program
    grounded_rules = [
        instantiate_rule_with_constants(rule, CONSTANT_POOL)
        for rule in logic_program.rules
    ]
    grounded_program = LogicProgram(grounded_rules)

    # Print the grounded logic program
    print("\nGrounded Logic Program:")
    print(grounded_program)

    # Generate graph for grounded logic program
    G = build_grounded_world_graph(grounded_program)
    draw_grounded_world_graph(G)


if __name__ == "__main__":
    main()

