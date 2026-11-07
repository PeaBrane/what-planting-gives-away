"""consistency of the (W4) window model with a real instance (premise (Q) of the construction notes): every model cell,
translated to each patch of the built torus instance, has the same type there (background / F6 with the same link
corners).  Usage: TILE_S=12 MARGIN=1 xcheck46.py NAME"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
import window46 as w  # noqa: E402
from tiles import DIR_TO_K  # noqa: E402
from verify_blocks import FORMULAS  # noqa: E402

S = B.S
rec_bg, rec_f6 = [], {}
_lock = w.lock


def lock(uf, cells):
    rec_bg.extend(cells)
    _lock(uf, cells)


def propagate(uf, anch, f6, links):
    rec_f6.update({c: frozenset(links[c]) for c in f6})
    return []


w.lock, w.propagate = lock, propagate
offs = w.giant_offsets()
w.W4(offs)
name = sys.argv[1]
margin = int(os.environ.get("MARGIN", "1"))
n_vars, clauses = FORMULAS[name]
origins, edges, paths = G.layout_A46(clauses)
lk, negs = G.expand46(clauses, origins, edges, paths)
lo = [min(c[j] for c in lk.used) for j in range(3)]
kshift = tuple(margin - math.floor(l / S) for l in lo)
L, netc, role = G.to_torus46(lk, negs, margin)
bad = {"bg_not_bg": 0, "f6_missing": 0, "f6_links_differ": 0}
for O in origins:
    sh = tuple(S * (O[j] + kshift[j]) for j in range(3))
    for c in rec_bg:
        p = tuple((c[j] + sh[j]) % L for j in range(3))
        if p in netc:
            bad["bg_not_bg"] += 1
    for c, ks in rec_f6.items():
        p = tuple((c[j] + sh[j]) % L for j in range(3))
        if p not in netc:
            bad["f6_missing"] += 1
        elif frozenset(netc[p].keys()) != ks:
            bad["f6_links_differ"] += 1
print(f"[{name}] L={L} patches={len(origins)}; W4 model: {len(set(rec_bg))} background cells, {len(rec_f6)} F6 cells;"
      f" mismatches per type summed over patches: {bad}")
