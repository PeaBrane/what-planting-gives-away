"""K_{4,r} checks against Bowlin (2012) and Claim 6.

mode 'bowlin8': Bowlin's extremal K_{4,8} (one column of each type in {+-1}^4 / +-): frustration index by brute
force, membership in P after gauging, rank of tight cuts (extreme ray of P = facet normal of CUT_c).
mode 'lp R': weighted sup f*_w(K_{4,R}) by an LP per multiset of column types (covers all switching classes up to
automorphism).
mode 'minors': K5-minor witnesses for Petersen and K_{4,4} by explicit contraction.
"""
import itertools
import sys
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

TYPES = [np.array((1,) + t) for t in itertools.product((1, -1), repeat=3)]  # tau[0] = +1


def bowlin8():
    l, r = 4, 8
    n = l + r
    edges = [(a, l + j) for a in range(l) for j in range(r)]
    tau = np.array([TYPES[j][a] for a in range(l) for j in range(r)])
    m = len(edges)
    # all s with s_0 = +1
    M = 1 << (n - 1)
    bits = ((np.arange(M)[:, None] >> np.arange(n - 1)[None, :]) & 1).astype(np.int8)
    S = np.ones((M, n), dtype=np.int8)
    S[:, 1:] = 1 - 2 * bits
    ea = np.array([e[0] for e in edges])
    eb = np.array([e[1] for e in edges])
    prod = S[:, ea] * S[:, eb]
    U = ((tau[None, :] * prod) < 0).sum(axis=1)
    k = int(np.argmin(U))
    print("Bowlin K_{4,8}: frustration index =", int(U.min()), " |E| =", m,
          " f =", Fraction(int(U.min()), m), " #ground states (s_0=+1) =", int((U == U.min()).sum()))
    s0 = S[k]
    Wg = tau * s0[ea] * s0[eb]
    cut = (prod < 0).astype(np.int64)
    cv = cut @ Wg
    print("  gauged W in P:", bool(cv.min() >= 0), " neg(W) =", int((Wg < 0).sum()))
    tight = cut[cv == 0].astype(float)
    print("  #tight cuts (incl. empty) =", len(tight), " rank =", np.linalg.matrix_rank(tight), " (m-1 =", m - 1, ")")


def lp_k4r(r):
    l = 4
    sA = [np.array((1,) + t) for t in itertools.product((1, -1), repeat=3)]
    best = (-1.0, None, None)
    count = 0
    for ms in itertools.combinations_with_replacement(range(8), r):
        count += 1
        nw = l * r
        nu = 8 * r
        nv = nw + nu + 1
        rows, rhs = [], []
        for si, s in enumerate(sA):
            row = np.zeros(nv)
            row[-1] = 1.0
            for j in range(r):
                row[nw + 8 * j + si] = -1.0
            rows.append(row)
            rhs.append(0.0)
            for j, ty in enumerate(ms):
                v = TYPES[ty] * s
                for side in (1, -1):
                    row = np.zeros(nv)
                    row[nw + 8 * j + si] = 1.0
                    for a in range(l):
                        if v[a] == side:
                            row[l * j + a] = -1.0
                    rows.append(row)
                    rhs.append(0.0)
        A_eq = np.zeros((1, nv))
        A_eq[0, :nw] = 1.0
        c = np.zeros(nv)
        c[-1] = -1.0
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * (nw + nu) + [(None, None)], method="highs")
        assert res.status == 0, res.message
        if -res.fun > best[0] + 1e-12:
            best = (-res.fun, ms, res.x[:nw].reshape(r, l))
    val, ms, w = best
    print(f"K_(4,{r}): multisets={count}  f*_w = {val:.6f}  ~ {Fraction(val).limit_denominator(200)}"
          f"  (5/16 = {5/16:.6f}, 1/4 = 0.25)")
    print("   best column types:", [tuple(TYPES[t]) for t in ms])
    print("   weights (rows = columns, scaled):", np.round(w / w.max(), 3).tolist())


def contract(n, edges, pairs):
    p = list(range(n))

    def f(x):
        while p[x] != x:
            x = p[x]
        return x

    for a, b in pairs:
        p[f(a)] = f(b)
    E = {tuple(sorted((f(a), f(b)))) for a, b in edges if f(a) != f(b)}
    V = {f(x) for x in range(n)}
    return V, E


def minors():
    pe = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)] + \
         [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    V, E = contract(10, pe, [(i, i + 5) for i in range(5)])
    print("Petersen / spokes contracted: |V| =", len(V), " |E| =", len(E), " K5:", len(V) == 5 and len(E) == 10)
    k44 = [(a, 4 + b) for a in range(4) for b in range(4)]
    V, E = contract(8, k44, [(0, 4), (1, 5), (2, 6)])
    print("K44 / 3 matching edges contracted: |V| =", len(V), " |E| =", len(E), " K5:", len(V) == 5 and len(E) == 10)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bowlin8":
        bowlin8()
    elif mode == "minors":
        minors()
    elif mode == "lp":
        for r in map(int, sys.argv[2:]):
            lp_k4r(r)
