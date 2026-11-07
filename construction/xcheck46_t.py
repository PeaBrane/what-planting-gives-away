"""consistency of the (W3) template window model with real instances: for every path block of the built torus
instance, the instance's F6 cells there are the model's template cells with the model's link corners, the block's
other interior cells and its foam are background.  Usage: TILE_S=12 MARGIN=1 xcheck46_t.py NAME"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
import templates46 as T6  # noqa: E402
from g46_local3 import block_foam, iscell  # noqa: E402
from layout import face_of  # noqa: E402
from tiles import DIR_TO_K  # noqa: E402
from verify_blocks import FORMULAS  # noqa: E402

S, H = B.S, B.H
name = sys.argv[1]
margin = int(os.environ.get("MARGIN", "1"))
n_vars, clauses = FORMULAS[name]
origins, edges, paths = G.layout_A46(clauses)
lk, negs = G.expand46(clauses, origins, edges, paths)
lo = [min(c[j] for c in lk.used) for j in range(3)]
kshift = tuple(margin - math.floor(l / S) for l in lo)
L, netc, role = G.to_torus46(lk, negs, margin)
T = T6.templates46()
bad = {"link_mismatch": 0, "interior_not_bg": 0, "foam_not_bg": 0}
nblk = 0
z1_nonsocket = 0
for path in paths:
    for t, g in enumerate(path):
        f1 = (2, -1) if t == 0 else face_of(B.vsub(path[t - 1], g))
        f2 = (2, -1) if t == len(path) - 1 else face_of(B.vsub(path[t + 1], g))
        if g[2] == 1 and (f1, f2) not in (((2, -1), (2, 1)), ((2, 1), (2, -1))):
            z1_nonsocket += 1
        nblk += 1
        C = tuple(S * x for x in g)
        cells_t = [B.vadd(C, c) for c in T[f1, f2]]
        idx = set(cells_t)
        sh = tuple(S * kshift[j] for j in range(3))
        P = lambda c: tuple((c[j] + sh[j]) % L for j in range(3))  # noqa: E731
        for i, c in enumerate(cells_t):
            ks = {DIR_TO_K[B.vsub(cells_t[i - 1], c)] if i > 0 else DIR_TO_K[B.neg(B.cross_step(f1, B.entry_tau(f1)))],
                  DIR_TO_K[B.vsub(cells_t[i + 1], c)] if i < len(cells_t) - 1 else DIR_TO_K[B.cross_step(f2, f2[1])]}
            if set(netc.get(P(c), {}).keys()) != ks:
                bad["link_mismatch"] += 1
        for d in itertools.product(range(-H + 1, H - 1), repeat=3):
            u = B.vadd(C, d)
            if iscell(u) and u not in idx and P(u) in netc:
                bad["interior_not_bg"] += 1
        for u in block_foam(g):
            if P(u) in netc:
                bad["foam_not_bg"] += 1
print(f"[{name}] L={L}: {nblk} path blocks checked against the (W3)/(W4) template model; z=1 non-socket templates:"
      f" {z1_nonsocket}; min path z = {min(g[2] for p in paths for g in p)}; mismatches {bad}")
