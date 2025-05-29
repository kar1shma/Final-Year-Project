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

# Consts & Logging

VAR_NAMES: List[str] = list("XYZWABCDE")
RETRY_LIMIT: int = 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Primitives

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

# Core helpers

def generate_logic_program(config: Dict[str,Any]) -> LogicProgram:
    """
    Generate a random LogicProgram of ground (variable-free) Horn rules according to config.
    All terms are constants to ensure safe grounding.
    """
    rules: List[Rule] = []
    usage: Dict[Predicate,int] = {}
    for _ in range(config.get("num_rules",0)):
        # choose head predicate
        if usage and random.random() < 0.5:
            cands = [p for p,c in usage.items() if c < config.get("branching_factor",1)]
            head_pred = random.choice(cands) if cands else random.choice(PREDICATE_POOL)
        else:
            head_pred = random.choice(PREDICATE_POOL)
        # ground head terms
        head_terms: List[Term] = []
        for typ in head_pred.arg_types:
            choices = CONSTANT_POOL.get(typ, [])
            const = random.choice(choices) if choices else random.choice(sum(CONSTANT_POOL.values(), []))
            head_terms.append(Term(const, False))
        head = Atom(head_pred, head_terms)
        # build body of ground atoms
        body: List[Atom] = []
        length = random.randint(1, config.get("max_body_length",1))
        for _ in range(length):
            pred = head_pred if config.get("allow_recursion", False) and random.random() < 0.25 else random.choice(PREDICATE_POOL)
            terms: List[Term] = []
            for typ in pred.arg_types:
                choices = CONSTANT_POOL.get(typ, [])
                const = random.choice(choices) if choices else random.choice(sum(CONSTANT_POOL.values(), []))
                terms.append(Term(const, False))
            body.append(Atom(pred, terms))
        rules.append(Rule(head, body))
        usage[head_pred] = usage.get(head_pred, 0) + 1
    return LogicProgram(rules)


def get_all_ground_atoms(program: LogicProgram)->Set[Atom]:
    atoms=set()
    preds={r.head.predicate for r in program.rules}|{b.predicate for r in program.rules for b in r.body}
    for p in preds:
        if p.arity==0:
            atoms.add(Atom(p,[]))
        elif p.arity==1:
            for c in CONSTANT_POOL.get(p.arg_types[0],[]): atoms.add(Atom(p,[Term(c,False)]))
        else:
            for c1 in CONSTANT_POOL.get(p.arg_types[0],[]):
                for c2 in CONSTANT_POOL.get(p.arg_types[1],[]):
                    if c1!=c2: atoms.add(Atom(p,[Term(c1,False),Term(c2,False)]))
    return atoms


def generate_base_facts(program: LogicProgram,num_facts:int)->List[Atom]:
    pool=list(get_all_ground_atoms(program))
    return random.sample(pool,min(num_facts,len(pool)))


def clingo_grounding(program:LogicProgram,base_facts:List[Atom])->Dict[str,Set[Atom]]:
    asp_rules=program.to_asp()
    asp_facts="\n".join(a.to_asp() for a in base_facts)
    preds={r.head.predicate for r in program.rules}|{b.predicate for r in program.rules for b in r.body}
    shows="\n".join(f"#show {p.name}/{p.arity}." for p in preds)
    ctl=clingo.Control(["--warn=none"])
    ctl.add("base",[],asp_rules+"\n"+asp_facts+"\n"+shows)
    ctl.ground([("base",[])])
    true_strs=set()
    ctl.solve(on_model=lambda m: true_strs.update(str(x) for x in m.symbols(shown=True)))
    true_atoms=set()
    for s in true_strs:
        if"("in s:
            n,args=s.split("(",1)
            args=args.rstrip(")").split(",")
            p=next((pp for pp in PREDICATE_POOL if pp.name==n and pp.arity==len(args)),None)
            if p: true_atoms.add(Atom(p,[Term(a,False) for a in args]))
        else:
            p=next((pp for pp in PREDICATE_POOL if pp.name==s and pp.arity==0),None)
            if p: true_atoms.add(Atom(p,[]))
    all_atoms=get_all_ground_atoms(program)
    return {"true_facts":true_atoms, "false_facts":all_atoms-true_atoms}


def determine_how_proved(program:LogicProgram,base_facts:List[Atom],target:Atom)->Dict[str,Any]:
    known,set_depth,der_rules=set(base_facts),{str(a):0 for a in base_facts},{}
    depth=0
    while True:
        new=set()
        for i,r in enumerate(program.rules):
            if r.is_fact(): continue
            for g in [r]:
                if all(b in known for b in g.body):
                    h=g.head
                    if h not in known:
                        new.add(h)
                        der_rules[str(h)]={"rule_idx":i,"body_atoms":[str(b) for b in g.body]}
                        set_depth[str(h)]=depth+1
        if not new: break
        depth+=1; known|=new
    if target not in known: return {"derivable":False,"steps":[],"depth":-1}
    steps=[]; cur=str(target)
    while cur in der_rules:
        info=der_rules[cur]; steps.append({"derived_fact":cur,**info})
        if not info["body_atoms"]: break
        cur=max(info["body_atoms"], key=lambda x:set_depth.get(x,0))
    steps.reverse()
    return {"derivable":True,"steps":steps,"depth":set_depth[str(target)]}


def explain_reasoning_steps(program:LogicProgram,base_facts:List[Atom],target:Atom,info:Dict[str,Any],true_facts:Set[Atom])->str:
    if not info.get("derivable"): return f"'{target.to_nl()}' cannot be derived."
    lines=["Reasoning process:"]+[f"- {a.to_nl()}" for a in base_facts]
    for i,st in enumerate(info["steps"],1):
        head=next(a for a in true_facts if str(a)==st["derived_fact"])  
        body=[next(a for a in true_facts if str(a)==b) for b in st["body_atoms"]]
        cond=", ".join(b.to_nl() for b in body)
        lines.append(f"Step {i}: Since {cond}, by Rule {st['rule_idx']+1} we get {head.to_nl()}.")
    lines.append(f"\nTherefore, '{target.to_nl()}' is true.")
    return "\n".join(lines)


# Inductive generation

def generate_inductive_program(config:Dict[str,Any])->Tuple[LogicProgram,List[Atom]]:
    """
    Build a LogicProgram guaranteed to have a proof chain of length `proof_depth` with ground atoms.
    Returns (program, base_facts)
    """
    d = config.get("proof_depth", 1)
    chain_rules: List[Rule] = []
    # Build a ground chain: each head uses constants, not variables
    prev_head: Optional[Atom] = None
    for i in range(d+1):  # chain length = proof_depth + 1
        # pick a predicate and ground terms
        p = random.choice(PREDICATE_POOL)
        terms: List[Term] = []
        for typ in p.arg_types:
            const = random.choice(CONSTANT_POOL.get(typ, []))
            terms.append(Term(const, False))
        head = Atom(p, terms)
        body = [prev_head] if prev_head is not None else []
        chain_rules.append(Rule(head, body))
        prev_head = head
    # Adjust remaining rules count
    remaining = max(config.get("num_rules", 0) - len(chain_rules), 0)
    rnd_cfg = {**config, "num_rules": remaining}
        # Generate the rest of the program
    rest_prog = generate_logic_program(rnd_cfg)
    # Separate chain root as a base fact to avoid 'fact rules'
    chain_rule_facts = chain_rules[:1]          # first rule is a ground fact
    chain_inference = chain_rules[1:]           # remaining chain for inference
    # Only include inference rules in the program
    program = LogicProgram(chain_inference + rest_prog.rules)
    # Base facts: include the root fact plus other random facts
    root_fact = chain_rule_facts[0].head
    other_base = generate_base_facts(program, max(config.get("num_base_facts", 1) - 1, 0))
    base_facts = [root_fact] + other_base
    return program, base_facts


# Single-case deduction helper

def _make_deduction_task(program:LogicProgram, base:List[Atom], config:Dict[str,Any], want_yes:bool) -> Tuple[str,str,str,str,Union[int,str]]:
    d = config["proof_depth"]
    chain_head = program.rules[d-1].head
    if want_yes:
        target, label, depth = chain_head, "YES", d
    else:
        pool = list(clingo_grounding(program, base)["false_facts"])
        if pool:
            target, label, depth = random.choice(pool), "NO", "not applicable"
        else:
            # fallback to a YES if no false atom found
            target, label, depth = chain_head, "YES", d
    q = target.to_asp().strip()
    c = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in base)
    prompt = generate_deduction_prompt(program, base, target)
    return q, c, prompt, label, depth


# Batch generators

def generate_multiple_deduction_tasks(program:LogicProgram, base:List[Atom], config:Dict[str,Any], n:int)->List[Dict[str,Any]]:
    tasks=[]
    for i in range(n):
        want_yes = i < n//2
        q, c, prompt, label, depth = _make_deduction_task(program, base, config, want_yes)
        tasks.append({"q":q,"c":c,"t":label,"prompt":prompt,
                      "metadata":{**config,"depth":depth,"reasoning_type":"deduction"}})
    return tasks


# Abduction single-case

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

def _make_abduction_task(program: LogicProgram,
                         base: List[Atom],
                         config: Dict[str,Any],
                         want_yes: bool
) -> Tuple[str, str, str, str, Union[int,str]]:
    """
    Build one abduction task:
      - Yes-case: hide a true leaf premise that is needed to prove obs.
      - No-case: hide some other base-fact.
    Returns (q, c, prompt, label, depth).
    """
    d = config["proof_depth"]
    # The observation is the last inference head
    obs = program.rules[d-1].head

    # Compute proof tree of obs
    info = determine_how_proved(program, base, obs)
    if not info["derivable"]:
        raise ValueError("Chain target not derivable")

    # Leaf premises are those facts that feed into the proof but are never themselves derived
    leaves = set(extract_abduction_hypotheses(info))
    # If depth=1 (just a fact), treat *all* base as leaves
    if want_yes and not leaves:
        leaves = {a.to_asp().rstrip('.') for a in base}

    # Decide which atom to hide (the hypothesis)
    if want_yes:
        candidates = leaves
    else:
        all_base = {a.to_asp().rstrip('.') for a in base}
        candidates = all_base - leaves

    if not candidates:
        # fallback: flip
        candidates = leaves if not want_yes else (set(a.to_asp().rstrip('.') for a in base) - leaves)

    hidden_str = random.choice(list(candidates))
    hidden = next(a for a in base if a.to_asp().rstrip('.') == hidden_str)

    # Build context without the hidden hypothesis — do *not* include OBS in context_facts
    context_facts = [a for a in base if a != hidden]

    # Serialize query and context
    q      = hidden.to_asp()
    c      = program.to_asp() + "\n" + "\n".join(a.to_asp() for a in context_facts)
    prompt = generate_abduction_prompt(program, context_facts, hidden, obs)

    label = "YES" if want_yes else "NO"
    depth = d if want_yes else "not applicable"
    return q, c, prompt, label, depth


# Abduction batch

def generate_multiple_abduction_tasks(program:LogicProgram, base:List[Atom], config:Dict[str,Any], n:int, yes_ratio:float=0.5)->List[Dict[str,Any]]:
    tasks=[]
    num_yes=int(n*yes_ratio)
    for i in range(n):
        want_yes=i<num_yes
        try: q,c,p,label,depth=_make_abduction_task(program,base,config,want_yes)
        except ValueError: q,c,p,label,depth=_make_abduction_task(program,base,config,not want_yes)
        tasks.append({"q":q,"c":c,"prompt":p,"t":label,
                      "metadata":{**config,"depth":depth,"reasoning_type":"abduction"}})
    return tasks


# LLM Prompts

def generate_deduction_prompt(program:LogicProgram, base:List[Atom], target:Atom)->str:
    # rules=[f" {i+1}: {r.to_nl()}" for i,r in enumerate(program.rules)]
    displayed = list(program.rules)
    random.shuffle(displayed)
    rules = [f" {i+1}: {r.to_nl()}" for i, r in enumerate(displayed)]
    facts=[f"- {a.to_nl()}" for a in base]
    return("You are given the following information:\n"+"\n".join(rules)
           +"\n\nAnd the following facts:\n"+"\n".join(facts)
           +f"\n\nQUESTION:\nIs {target.to_nl()} true?\nAnswer exactly YES or NO.")

def generate_abduction_prompt(program:LogicProgram, ctx:List[Atom], query:Atom, obs: Atom)->str:
    # rules=[f" {i+1}: {r.to_nl()}" for i,r in enumerate(program.rules)]
    displayed = list(program.rules)
    random.shuffle(displayed)
    rules = list(program.rules)
    random.shuffle(rules)
    rules_nl = [f" {i+1}: {r.to_nl()}" for i,r in enumerate(rules)]
    facts_nl = [f"- {a.to_nl()}." for a in ctx]
    return (
        "You are given the following rules:\n"
        + "\n".join(rules_nl)
        + "\n\nAnd the following facts:\n"
        + "\n".join(facts_nl)
        + f"\n\nOBSERVATION: {obs.to_nl()} is true."
        + f"\n\nQUESTION:\nCould {query.to_nl()} be true?"
        + "\nAnswer exactly YES or NO."
    )


# Main driver: sweeping diverse configs
# ---------------------------------------------
if __name__ == "__main__":
    from itertools import product
    # Full grid
    NUM_RULES      = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    NUM_BASE_FACTS = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    PROOF_DEPTHS   = [1, 2, 3, 4, 5]
    TASKS_PER_GROUP = 5  # reuse each context (rules + base facts) for 5 tasks
    GROUPS_PER_CFG = 2  # number of distinct contexts per config

    all_tasks = []
    for nr, nb, pd in product(NUM_RULES, NUM_BASE_FACTS, PROOF_DEPTHS):
        cfg = {
            "num_rules":       nr,
            "max_body_length": 3,
            "allow_recursion": True,
            "branching_factor": 2,
            "proof_depth":      pd,
            "num_base_facts":   nb,
        }
        for _ in range(GROUPS_PER_CFG):
            # generate one shared context
            prog, base = generate_inductive_program(cfg)
            # yield TASKS_PER_GROUP deduction and abduction each
            ded = generate_multiple_deduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
            abd = generate_multiple_abduction_tasks(prog, base, cfg, TASKS_PER_GROUP)
            all_tasks.extend(ded)
            all_tasks.extend(abd)

    # Save full benchmark
    with open("benchmark.json", "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"Generated {len(all_tasks)} total tasks in benchmark.json")
