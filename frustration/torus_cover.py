"""Claim 7: one-edge-per-plaquette sets X on the L^3 periodic cubic lattice (multigraph for L = 2).

Edge (r, mu) joins r and r + e_mu.  Plaquette (r, mu<nu) = {(r,mu), (r+e_nu,mu), (r,nu), (r+e_mu,nu)}.
Formula (all even L):  x-edges with (y,z) = (0,0) mod 2,  y-edges with (x,z) = (0,1) mod 2,
                       z-edges with (x,y) = (1,1) mod 2.
"""
import itertools

import numpy as np


def eid(L, r, mu):
    x, y, z = (c % L for c in r)
    return ((x * L + y) * L + z) * 3 + mu


def build(L):
    E = 3 * L ** 3
    plaq = []
    for r in itertools.product(range(L), repeat=3):
        for mu, nu in ((0, 1), (0, 2), (1, 2)):
            em = [0, 0, 0]
            en = [0, 0, 0]
            em[mu] = 1
            en[nu] = 1
            rp_nu = tuple(r[i] + en[i] for i in range(3))
            rp_mu = tuple(r[i] + em[i] for i in range(3))
            plaq.append((eid(L, r, mu), eid(L, rp_nu, mu), eid(L, r, nu), eid(L, rp_mu, nu)))
    return E, np.array(plaq)


def formula(L):
    X = np.zeros(3 * L ** 3, dtype=bool)
    for x, y, z in itertools.product(range(L), repeat=3):
        X[eid(L, (x, y, z), 0)] = (y % 2 == 0 and z % 2 == 0)
        X[eid(L, (x, y, z), 1)] = (x % 2 == 0 and z % 2 == 1)
        X[eid(L, (x, y, z), 2)] = (x % 2 == 1 and y % 2 == 1)
    return X


def endpoints(L):
    a, b = [], []
    for x, y, z in itertools.product(range(L), repeat=3):
        for mu in range(3):
            r = [x, y, z]
            s = r.copy()
            s[mu] = (s[mu] + 1) % L
            a.append((x * L + y) * L + z)
            b.append((s[0] * L + s[1]) * L + s[2])
    return np.array(a), np.array(b)


for L in (2, 4, 6):
    E, P = build(L)
    assert all(len(set(p)) == 4 for p in P), "plaquette with repeated edge"
    inc = np.zeros(E, dtype=int)
    for p in P:
        inc[p] += 1
    distinct_plaq_per_edge = {len({tuple(sorted(p)) for p in P if e in p}) for e in range(E)} if L == 2 else None
    X = formula(L)
    per_plaq = X[P].sum(axis=1)
    J = np.where(X, -1, 1)
    frustrated = (np.prod(J[P], axis=1) == -1).all()
    print(f"L={L}: |E|={E} #plaq={len(P)} edges-per-plaquette-in-X: min={per_plaq.min()} max={per_plaq.max()} "
          f"|X|={X.sum()} |E|/4={E // 4}  every edge in 4 plaquettes: {bool((inc == 4).all())}"
          f"  all plaquettes frustrated: {bool(frustrated)}  U(+1)={X.sum()}  lower bound #plaq/4={len(P) // 4}"
          + (f"  distinct plaquettes per edge (L=2): {distinct_plaq_per_edge}" if L == 2 else ""))
    if L == 2:
        a, b = endpoints(L)
        n = L ** 3
        Smat = 1 - 2 * ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
        unsat = ((J[None, :] * Smat[:, a] * Smat[:, b]) < 0).sum(axis=1)
        print(f"   L=2 brute force over 2^{n} states: min U = {unsat.min()}  (|E|/4 = {E // 4})")
        # parallel-edge 2-cycles: both copies in X or both out?
        par_ok = all(X[eid(L, (x, y, z), mu)] == X[eid(L, tuple((c + (i == mu)) for i, c in enumerate((x, y, z))), mu)]
                     for x, y, z in itertools.product(range(L), repeat=3) for mu in range(3))
        print(f"   L=2 parallel pairs have equal sign (2-cycles unfrustrated): {par_ok}")
        # count all exact covers for L=2 by brute force over 6-subsets
        Pm = np.zeros((len(P), E), dtype=np.int8)
        for k, p in enumerate(P):
            Pm[k, p] = 1
        cnt = 0
        for comb in itertools.combinations(range(E), E // 4):
            if (Pm[:, comb].sum(axis=1) == 1).all():
                cnt += 1
        print(f"   L=2 number of exact one-per-plaquette sets: {cnt}")
