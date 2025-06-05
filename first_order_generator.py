# ──────────────────────────────────────────────────────────────────────────
# first_order_generator.py   (2025-06-05)
# Generates deduction + abduction benchmarks for first-order Horn programs.
# ──────────────────────────────────────────────────────────────────────────
import random, json, clingo
from itertools import product
from typing import List, Dict, Any, Optional, Tuple, Set

# -------------------------------------------------------------------------
# 1.  Core data structures
# -------------------------------------------------------------------------
class Predicate:
    def __init__(self, name: str, arity: int,
                 arg_types: Optional[List[str]] = None,
                 nl_template: Optional[str] = None):
        self.name, self.arity = name, arity
        self.arg_types = arg_types or ["any"] * arity
        self.nl_template = nl_template or self._default_template()
    def _default_template(self):
        if self.arity == 0: return f"it is {self.name}"
        if self.arity == 1: return f"{{0}} is {self.name}"
        return f"{self.name}(" + ", ".join(f"{{{i}}}" for i in range(self.arity)) + ")"
    def __repr__(self): return f"{self.name}/{self.arity}"

class Atom:
    def __init__(self, predicate: "Predicate", terms: List[str]):
        assert len(terms) == predicate.arity, "arity mismatch"
        self.predicate, self.terms = predicate, terms
    def __repr__(self):
        return self.predicate.name if self.predicate.arity == 0 \
               else f"{self.predicate.name}({','.join(self.terms)})"
    def to_asp(self): return f"{self}."
    def to_nl(self):
        repl = []
        for i,t in enumerate(self.terms):
            if t[0].isupper():
                typ = self.predicate.arg_types[i]
                repl.append("someone" if typ == "person" else "something")
            else: repl.append(t)
        return self.predicate.nl_template.format(*repl)
    def __hash__(self): return hash(repr(self))
    def __eq__(self, other): return repr(self) == repr(other)

class Rule:
    def __init__(self, head: Atom, body: Optional[List[Atom]]=None):
        self.head, self.body = head, body or []
    def is_fact(self): return not self.body
    def __repr__(self):
        return f"{self.head}." if self.is_fact() \
               else f"{self.head} :- {', '.join(map(str,self.body))}."
    def to_asp(self): return repr(self)
    def to_nl(self):
        head_nl = self.head.to_nl()
        if self.is_fact(): return head_nl + "."
        body_nl = " and ".join(b.to_nl() for b in self.body)
        return f"If {body_nl}, then {head_nl}."

class LogicProgram:
    def __init__(self, rules: Optional[List[Rule]]=None):
        self.rules = rules or []
    def add(self, r: Rule): self.rules.append(r)
    def __iter__(self): return iter(self.rules)
    def to_asp(self): return "\n".join(r.to_asp() for r in self.rules)

# -------------------------------------------------------------------------
# 2.  Predicate & constant pools
# -------------------------------------------------------------------------
PREDICATE_POOL = [
    Predicate("sunny",0,[], "it is sunny"),
    Predicate("cold",0,[],  "it is cold"),
    Predicate("hot",0,[],   "it is hot"),
    Predicate("wet",1,["object"],   "{0} is wet"),
    Predicate("big",1,["object"],   "{0} is big"),
    Predicate("small",1,["object"], "{0} is small"),
    Predicate("sad",1,["person"],   "{0} is sad"),
    Predicate("happy",1,["person"], "{0} is happy"),
    Predicate("hungry",1,["person"],"{0} is hungry"),
    Predicate("owns",2,["person","object"], "{0} owns {1}"),
    Predicate("likes",2,["person","object"],"{0} likes {1}"),
    Predicate("dislikes",2,["person","object"],"{0} dislikes {1}"),
    Predicate("friend",2,["person","person"], "{0} is a friend of {1}"),
    Predicate("enemy",2,["person","person"],  "{0} is an enemy of {1}"),
    Predicate("parent",2,["person","person"], "{0} is a parent of {1}"),
    Predicate("sibling",2,["person","person"],"{0} is a sibling of {1}"),
]
CONSTANT_POOL: Dict[str,List[str]] = {
    "person": ["alice","bob","carol","dave","eve","frank","george"],
    "object": ["apple","book","ball","car","pencil","phone"],
}
VAR_POOL = ["X","Y","Z","U","V","W"]

# -------------------------------------------------------------------------
# 3.  Grounding helpers
# -------------------------------------------------------------------------
def ground_rule(r: Rule) -> List[Rule]:
    """Return all ground instances of a rule (unsafe vars allowed here)."""
    vars={t for a in [r.head]+r.body for t in a.terms if t[0].isupper()}
    if not vars: return [r]
    v_types={}
    for a in [r.head]+r.body:
        for i,t in enumerate(a.terms):
            if t[0].isupper(): v_types[t]=a.predicate.arg_types[i]
    ordered=sorted(vars)
    domains=[CONSTANT_POOL[v_types[v]] for v in ordered]
    out=[]
    for tpl in product(*domains):
        m=dict(zip(ordered, tpl))
        def inst(atom: Atom):
            return Atom(atom.predicate, [m.get(t,t) for t in atom.terms])
        out.append(Rule(inst(r.head), [inst(b) for b in r.body]))
    return out

def ground_program(prog: LogicProgram)->LogicProgram:
    out=[]
    for r in prog: out.extend(ground_rule(r))
    return LogicProgram(out)

# -------------------------------------------------------------------------
# 4.  ASP solving
# -------------------------------------------------------------------------
def solve_clingo(prog: LogicProgram, facts: List[Atom])->Set[Atom]:
    text = prog.to_asp()+"\n"+"\n".join(a.to_asp() for a in facts)
    preds={r.head.predicate for r in prog}
    preds|={b.predicate for r in prog for b in r.body}
    text += "\n" + "\n".join(f"#show {p.name}/{p.arity}." for p in preds)
    ctl=clingo.Control(["--warn=none"])
    ctl.add("base",[],text); ctl.ground([("base",[])])
    truth=set()
    ctl.solve(on_model=lambda m: truth.update(str(x) for x in m.symbols(shown=True)))
    out=set()
    for s in truth:
        if "(" in s: name,rest=s.split("(",1); args=rest[:-1].split(",")
        else: name,args=s,[]
        pred=next(p for p in PREDICATE_POOL if p.name==name and p.arity==len(args))
        out.add(Atom(pred,args))
    return out

# -------------------------------------------------------------------------
# 5.  Program generator
# -------------------------------------------------------------------------
def make_inductive_program(cfg:Dict[str,Any]) -> Tuple[LogicProgram,List[Atom]]:
    d=cfg["proof_depth"]; prog=LogicProgram()
    var="X"
    unary=[p for p in PREDICATE_POOL if p.arity==1]
    head_pred=random.choice(unary)
    prog.add(Rule(Atom(head_pred,[var]),[]))
    prev=head_pred
    for _ in range(d):
        new_pred=random.choice(unary)
        prog.add(Rule(Atom(new_pred,[var]),[Atom(prev,[var])]))
        prev=new_pred
    const=random.choice(CONSTANT_POOL[head_pred.arg_types[0]])
    root_fact=Atom(head_pred,[const])

    # extra safe rules
    extra=max(cfg["num_rules"]-(d+1),0)
    var_ctr=0
    for _ in range(extra):
        p=random.choice(PREDICATE_POOL)
        head_vars=[VAR_POOL[(var_ctr+i) % len(VAR_POOL)] for i in range(p.arity)]
        var_ctr+=p.arity
        body=[]
        length=random.randint(1,cfg["max_body_length"])
        for _ in range(length):
            pb=random.choice(PREDICATE_POOL)
            terms=[]
            for t in pb.arg_types:
                reuse=[v for v,ty in zip(head_vars,p.arg_types) if ty==t]
                if reuse and random.random()<0.5: terms.append(random.choice(reuse))
                else:
                    terms.append(VAR_POOL[var_ctr % len(VAR_POOL)])
                    var_ctr+=1
            body.append(Atom(pb,terms))
        # safety: each head var appears somewhere in body
        if all(any(v in b.terms for b in body) for v in head_vars):
            prog.add(Rule(Atom(p,head_vars), body))
    # gather ground atoms
    gprog=ground_program(prog)
    atoms=set()
    for r in gprog: atoms.add(r.head); atoms.update(r.body)
    others=[a for a in atoms if a!=root_fact]
    base=[root_fact]+random.sample(others, min(cfg["num_base_facts"]-1,len(others)))
    return prog,base

# -------------------------------------------------------------------------
# 6.  Safe sampler
# -------------------------------------------------------------------------
def safe_pick(pop: Set[Atom], k: int)->List[Atom]:
    pop=list(pop)
    if not pop: return []
    if k<=len(pop): return random.sample(pop,k)
    return pop + random.choices(pop,k=k-len(pop))

# -------------------------------------------------------------------------
# 7.  Deduction tasks
# -------------------------------------------------------------------------
def make_deduction_tasks(prog:LogicProgram, base:List[Atom],
                         cfg:Dict[str,Any], n:int)->List[Dict[str,Any]]:
    truth=solve_clingo(ground_program(prog),base)
    all_atoms={r.head for r in ground_program(prog)}
    all_atoms|={b for r in ground_program(prog) for b in r.body}
    false=all_atoms-truth
    yes=safe_pick(truth, int(n*0.6))
    no =safe_pick(false,  n-len(yes))
    tasks=[]
    for tgt in yes+no:
        label="true" if tgt in yes else "false"
        q=tgt.to_asp().strip()
        c=prog.to_asp()+"\n"+"\n".join(a.to_asp() for a in base)
        nl="You are given the following first-order rules:\n"+\
            "\n".join(r.to_nl() for r in prog.rules)+\
            "\n\nAnd the following ground facts:\n"+\
            "\n".join(a.to_nl()+"." for a in base)+\
            f"\n\nQUESTION:\nIs {tgt.to_nl()} true?\nAnswer exactly true or false."
        tasks.append({"q":q,"c":c,"natural language":nl,"t":label,
                      "metadata":{**cfg,"depth":cfg["proof_depth"],
                                  "reasoning_type":"deduction"}})
    return tasks

# -------------------------------------------------------------------------
# 8.  Abduction tasks  (hide-one-fact variant)
# -------------------------------------------------------------------------
def make_abduction_tasks(prog:LogicProgram, base:List[Atom],
                         cfg:Dict[str,Any], n:int)->List[Dict[str,Any]]:
    gprog=ground_program(prog)
    yes_pool,no_pool=set(),set()
    for f in base:
        rem=[b for b in base if b!=f]
        derivable = f in solve_clingo(gprog, rem)
        (yes_pool if derivable else no_pool).add(f)
    yes=safe_pick(yes_pool, int(n*0.6))
    no =safe_pick(no_pool,  n-len(yes))
    tasks=[]
    for hid in yes+no:
        remain=[b for b in base if b!=hid]
        label="true" if hid in yes else "false"
        q=hid.to_asp().strip()
        c=prog.to_asp()+"\n"+"\n".join(a.to_asp() for a in remain)
        nl="You are given the following first-order rules:\n"+\
            "\n".join(r.to_nl() for r in prog.rules)+\
            "\n\nAnd the following facts (except one hidden):\n"+\
            "\n".join(a.to_nl()+"." for a in remain)+\
            f"\n\nQUESTION:\nCould {hid.to_nl()} be true?\nAnswer exactly true or false."
        tasks.append({"q":q,"c":c,"natural language":nl,"t":label,
                      "metadata":{**cfg,"depth":"n/a","reasoning_type":"abduction"}})
    return tasks

# -------------------------------------------------------------------------
# 9.  Driver
# -------------------------------------------------------------------------
if __name__ == "__main__":
    PARAMS=dict(
        NUM_RULES=[5,15],
        NUM_BASE_FACTS=[3,6],
        PROOF_DEPTHS=[1,3],
        RECURSION=[True,False],
        TASKS_PER=5,
        GROUPS=1,
    )
    tasks=[]
    for nr in PARAMS["NUM_RULES"]:
      for nb in PARAMS["NUM_BASE_FACTS"]:
        for pd in PARAMS["PROOF_DEPTHS"]:
          for rec in PARAMS["RECURSION"]:
            if pd>nr: continue
            cfg=dict(num_rules=nr,max_body_length=3,
                     allow_recursion=rec,branching_factor=2,
                     proof_depth=pd,num_base_facts=nb)
            for _ in range(PARAMS["GROUPS"]):
                prog,base=make_inductive_program(cfg)
                tasks+=make_deduction_tasks(prog,base,cfg,PARAMS["TASKS_PER"])
                tasks+=make_abduction_tasks(prog,base,cfg,PARAMS["TASKS_PER"])
    with open("first_order_benchmark.json","w") as f:
        json.dump(tasks,f,indent=2)
    print(f"Generated {len(tasks)} tasks → first_order_benchmark.json")
