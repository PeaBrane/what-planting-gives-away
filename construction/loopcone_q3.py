"""Machine check behind Corollary 4: every +-1 cube coupling with the all-plus state among its minimisers lies in
the loop cone K(Q3) = cone{l_{C,f}} + R_{>=0}^E (LP feasibility over all cycles of Q3), independently of (Sey)."""
import itertools
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import CUBE_E, gs_set  # noqa: E402

E = len(CUBE_E)
adj = {k: [] for k in range(8)}
for i, (a, b) in enumerate(CUBE_E):
    adj[a].append((b, i))
    adj[b].append((a, i))
cycles = set()


def dfs(start, v, seen, edges):
    for w, i in adj[v]:
        if w == start and len(edges) >= 3:
            cycles.add(frozenset(edges + [i]))
        elif w not in seen and w > start:
            dfs(start, w, seen | {w}, edges + [i])


for s in range(8):
    dfs(s, s, {s}, [])
cycles = [c for c in cycles if len(c) >= 4]
atoms = []
for c in cycles:
    for f in c:
        v = np.zeros(E)
        for e in c:
            v[e] = 1.0
        v[f] = -1.0
        atoms.append(v)
A = np.array(atoms + [np.eye(E)[i] for i in range(E)]).T  # E x (atoms + E)
planted = set()
for W in itertools.product([1, -1], repeat=E):
    G, _ = gs_set(W)
    if 0 in G:
        planted.add(W)
bad = 0
for W in planted:
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=np.array(W, float), bounds=(0, None), method="highs")
    bad += r.status != 0
print(f"Q3 cycles: {len(cycles)} (by length {sorted({len(c) for c in cycles})}), loop atoms: {len(atoms)};"
      f" +-1 couplings with all-plus a ground state: {len(planted)}; not in K(Q3): {bad}")
