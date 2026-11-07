"""Exact integer checks of two weighted examples used in Section 5 of the paper (Example 5.4)."""
import itertools
from fractions import Fraction
import numpy as np

def check(name, n, edges, W):
    W = np.array(W)
    best = None
    cuts = []
    for s in itertools.product((1, -1), repeat=n - 1):
        s = (1,) + s
        U = sum(abs(w) for (a, b), w in zip(edges, W) if w * s[a] * s[b] < 0)
        best = U if best is None else min(best, U)
        cuts.append([int(s[a] * s[b] < 0) for a, b in edges])
    cuts = np.array(cuts)
    cv = cuts @ W
    tight = cuts[cv == 0]
    print(f"{name}: min U = {best}, |W| = {abs(W).sum()}, f = {Fraction(int(best), int(abs(W).sum()))}, "
          f"U(+1) = {int(-W[W < 0].sum())}, +1 ground state (W in P): {bool(cv.min() >= 0)}, "
          f"tight-cut rank = {np.linalg.matrix_rank(tight.astype(float))} of m-1 = {len(edges) - 1}")

# diamond K4-e: triangles (0,1,2) and (1,2,3) share edge s = (1,2); full-support weights 1,1,2(neg),1,1
check("diamond full support", 4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], [1, 1, -2, 1, 1])
# K44 LP optimum scaled by 2 (edge order (a, 4+b), a,b in 0..3)
k44 = [(a, 4 + b) for a in range(4) for b in range(4)]
check("K44 LP facet normal", 8, k44, [-1, 2, 2, 1, 2, 1, -1, 2, 2, -1, 2, -1, -1, -2, 1, 2])
# K5 pentagonal normal r_pq = -b_p b_q, b = (1,1,1,-1,-1)
b = (1, 1, 1, -1, -1)
k5 = [(p, q) for p in range(5) for q in range(p + 1, 5)]
check("K5 pentagonal", 5, k5, [-b[p] * b[q] for p, q in k5])
