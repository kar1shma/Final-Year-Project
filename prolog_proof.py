from pyswip import Prolog, Atom, Functor, Variable, Query

# A small meta-interpreter in Prolog for extracting proof trees
META_INTERPRETER = """
% Known facts
:- dynamic known/1.

% Rule representation: rule(ID, Head, BodyList).
:- dynamic rule/3.

% prove/2: proves a goal and returns a proof tree
prove(Goal, fact(Goal)) :- known(Goal).
prove(Goal, rule(ID, SubProofs)) :- rule(ID, Goal, Body), prove_list(Body, SubProofs).

% prove_list/2: prove each element in a list
prove_list([], []).
prove_list([H|T], [P|Ps]) :- prove(H, P), prove_list(T, Ps).
"""

def extract_proof_tree_with_prolog(rules, facts, query):
    """
    Uses SWI-Prolog via pyswip to build a proof tree for `query`.
    Returns a nested Python dict representing the tree, or None if not derivable.
    """
    prolog = Prolog()
    # Load meta-interpreter
    prolog.assertz('use_module(library(lists))')
    for line in META_INTERPRETER.splitlines():
        if line.strip():
            prolog.assertz(line)

    # Assert known facts
    for f in facts:
        atom = f.rstrip('.')
        prolog.assertz(f"known({atom})")

    # Assert rules with unique IDs
    for idx, r in enumerate(rules):
        head = r.head.to_asp().rstrip('.')
        body_atoms = [b.to_asp().rstrip('.') for b in r.body]
        body_list = '[' + ','.join(body_atoms) + ']'  # e.g. [sunny,wet(ball)]
        prolog.assertz(f"rule({idx}, {head}, {body_list})")

    # Query the proof tree
    Tree = Variable()
    q = Query(Functor('prove', 2)(Atom(query.predicate.name, *[Atom(t.name) for t in query.terms]), Tree))
    result = None
    if q.nextSolution():
        result = _prolog_tree_to_python(Tree.value)
    q.close()
    return result


def _prolog_tree_to_python(node):
    """
    Recursively convert Prolog proof tree terms into Python dicts.
    node is a Functor: either fact(Atom) or rule(ID, SubProofs)
    """
    name = node.name
    args = node.args
    if name == 'fact':
        return {'type':'fact', 'goal': str(args[0])}
    elif name == 'rule':
        rid = int(str(args[0]))
        sub = args[1]
        # sub is a Prolog list of proof nodes
        subs = []
        for elem in sub:
            subs.append(_prolog_tree_to_python(elem))
        return {'type':'rule', 'rule_idx': rid, 'subproofs': subs}
    else:
        # unknown structure
        return {'type':'unknown', 'repr': str(node)}


# from generator import ground, generate_logic_program, build_seed_facts, CONSTANT_POOL
# gp = generate_logic_program()
# gp_grounded = ground(gp, CONSTANT_POOL)
# facts, query = build_seed_facts(gp_grounded, min_depth=2)
# tree = extract_proof_tree_with_prolog(gp_grounded, facts, query)
# print(tree)
