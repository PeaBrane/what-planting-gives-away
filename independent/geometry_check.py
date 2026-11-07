"""Independent checks of Sec. 4.1-4.3 (Lemmas 4.1-4.5) with independent code; construction/ is used only to obtain the
template and cluster objects (B.templates(), B.cluster_template()), which are compared with templates_out.txt."""
import os
import ast
import itertools
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "construction"))
import blocks as B  # noqa: E402

STEPS = list(itertools.product((-1, 1), repeat=3))
TP = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def pi(a):
    return a[0] * a[1] * a[2]


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def e(i, s=1):
    return tuple(s if j == i else 0 for j in range(3))


log = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


# Lemma 4.1 / 4.2 / polarity by enumeration
l41 = all((sum(x != y for x, y in zip([-v for v in a], b)) == 1) == (dot(a, b) == -1) for a in STEPS for b in STEPS)
l42 = all(pi(a) == pi(b) for a in STEPS for b in STEPS if dot(a, b) == -1)
alt = all(pi(a) != pi(b) for a in STEPS for b in STEPS if sum(x != y for x, y in zip(a, b)) == 1)
p("Lemma 4.1 (domino iff a.a' = -1):", l41, "| Lemma 4.2 (pi preserved):", l42, "| face corners alternate in pi:", alt)
p("T+ = {a : pi(a) = +1}:", sorted(TP) == sorted(a for a in STEPS if pi(a) == 1))


# Boundary pairs as specified for the construction, S = 8
S, H = 8, 4


def a_canon(i, tau):
    return next(a for a in TP if a[i] == tau)


def aperp(i, tau):
    a = a_canon(i, tau)
    return tuple(0 if j == i else a[j] for j in range(3))


def bpair(C, i, sg, tau):
    ap = aperp(i, tau)
    if sg == 1:
        return add(add(C, ap), e(i, 3)), add(C, e(i, 2))
    return add(C, e(i, -4)), add(add(C, ap), e(i, -3))


def in_block(c, C=(0, 0, 0)):
    return all(-H <= c[j] - C[j] < H for j in range(3))


def is_cell(c):
    return c[0] % 2 == c[1] % 2 == c[2] % 2


# gluing (Lemma 4.4): exit outer of B(0) through (i, s) for travel s, entry outer of B(s e_i) for travel s
glue_ok = True
for i in range(3):
    for s in (1, -1):
        o_exit, i_exit = bpair((0, 0, 0), i, s, s)
        Cn = e(i, S * s)
        o_ent, i_ent = bpair(Cn, i, -s, s)
        steps = [sub(o_exit, i_exit), sub(o_ent, o_exit), sub(i_ent, o_ent)]
        ok = all(a in TP for a in steps) and all(dot(a, b) == -1 for a, b in zip(steps, steps[1:]))
        ok &= in_block(o_exit) and in_block(i_exit) and in_block(o_ent, Cn) and in_block(i_ent, Cn)
        glue_ok &= ok
p("Lemma 4.4 gluing, all 6 (face, travel) cases, steps in T+ and domino rule at both outer cells:", glue_ok)

# Templates (Lemma 4.3): compare live object with templates_out.txt, then check
txt = open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "construction"), "templates_out.txt")).read()
file_t = {}
for m_ in re.finditer(r"in (\(\d, -?1\)) out (\(\d, -?1\)): (\d+) cells: (\[.*?\])\n", txt):
    file_t[ast.literal_eval(m_.group(1)), ast.literal_eval(m_.group(2))] = ast.literal_eval(m_.group(4))
live = B.templates()
p("templates in file:", len(file_t), "live:", len(live), "identical:", file_t == {k: list(v) for k, v in live.items()})
tmpl_ok, sizes, contacts = True, [], {}
for (f1, f2), cells in sorted(file_t.items()):
    (i1, s1), (i2, s2) = f1, f2
    t_in, t_out = -s1, s2
    ok = f1 != f2 and all(in_block(c) and is_cell(c) for c in cells) and len(set(cells)) == len(cells)
    ok &= (cells[0], cells[1]) == bpair((0, 0, 0), i1, s1, t_in)
    ok &= (cells[-1], cells[-2]) == bpair((0, 0, 0), i2, s2, t_out)
    # across-face steps: predecessor = exit outer of the neighbouring block for travel t_in
    Cprev = e(i1, S * s1)
    prev = bpair(Cprev, i1, -s1, t_in)[0]
    Cnext = e(i2, S * s2)
    nxt = bpair(Cnext, i2, -s2, t_out)[0]
    seq = [prev] + cells + [nxt]
    st = [sub(b, a) for a, b in zip(seq, seq[1:])]
    ok &= all(a in TP for a in st) and all(dot(a, b) == -1 for a, b in zip(st, st[1:]))
    tmpl_ok &= ok
    sizes.append(len(cells))
    # non-consecutive vertex-sharing pairs (contacts) inside the template
    cs = [(a, b) for a in range(len(cells)) for b in range(a + 2, len(cells))
          if all(abs(x - y) == 1 for x, y in zip(cells[a], cells[b]))]
    contacts[f1, f2] = [(a, b) for a, b in cs]
p("Lemma 4.3: all 30 templates in block, simple, canonical boundary pairs, T+ steps, domino rule incl. across faces:",
  tmpl_ok, "| sizes", min(sizes), "-", max(sizes))
p("internal contacts (index pairs) per template, nonzero only:",
  {k: v for k, v in contacts.items() if v})
p("(observation, not a T claim) all internal contacts are (t, t+3):", all(b - a == 3 for v in contacts.values() for a, b in v))

# Cluster template (Lemma 4.5)
lk, assign, wires, u = B.cluster_template()
R = B.cluster_region((0, 0, 0))
p("cluster: u =", u, "cells", len(lk.used), "region", R, "all in region:",
  all(all(lo <= c[j] < hi for j, (lo, hi) in enumerate(R)) for c in lk.used))
p("socket assignment (literal pos, polarity) -> (a, b):", assign,
  "| matches doc (a = i, b = 1 for +, 0 for -):", all(assign[i, s] == (i, 1 if s == 1 else 0) for i in range(3) for s in (1, -1)))
# independent shape check of the cluster: link directions per cell
shape_ok = True
outer_cells = {w[-1] for w in wires.values()}
bad_cells = []
for c in lk.used:
    dirs = list(lk.net[c].keys())
    role = lk.role[c]
    nets = set(lk.net[c].values())
    if role.startswith("clause"):
        shape_ok &= sorted(dirs) == sorted(STEPS)
        continue
    if len(dirs) == 2:
        a, b = dirs
        shape_ok &= dot(a, b) == 1 and len(nets) == 1            # corner directions adjacent = domino
    elif len(dirs) == 4:
        shape_ok &= any(len({d[ax] for d in dirs}) == 1 for ax in range(3)) and len(nets) == 1
    elif len(dirs) == 1 and c in outer_cells:
        pass                                          # socket outer cell: its second link is made in expand()
    else:
        shape_ok = False
        bad_cells.append((c, role, dirs))
p("bad cluster cells:", bad_cells[:5])
p("cluster link shapes (clause 8 / domino / face, one network per non-clause cell):", shape_ok)
# polarity of ports and socket travel directions
pol_ok = True
for (i, s), w in wires.items():
    q = w[0]
    outer = w[-1]
    a_b = (assign[i, s])
    C0 = (S * 2 * a_b[0], S * 2 * a_b[1], 0)
    pol_ok &= (outer, w[-2]) == bpair(C0, 2, 1, s)
    st = [sub(b, a) for a, b in zip(w, w[1:])]
    # port wire read outward from Q: T_+ for + ports, T_- for - ports
    pol_ok &= all(pi(a) == s for a in st)
p("port wires end at the +z boundary pair of block (2a, 2b, 0) for travel = polarity, chirality = polarity:", pol_ok)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry_check_out.txt"), "w") as fh:
    fh.write("\n".join(log) + "\n")
