"""Is every planted RBM (gauged so +1 is a ground state) a conical combination of
4-cycle loop atoms (+1,+1,+1,-1) plus a non-negative matrix?  (Pei-Manukian-Di Ventra,
arXiv:1905.05334, Sec. IV.F, statement (7).)

Prediction from Seymour (cut cone = metric cone iff no K5 minor): yes for min(n,m) <= 3,
no in general for n, m >= 4.
"""
import itertools
import sys

import numpy as np
from scipy.optimize import linprog


def edges(n, m):
    return [(i, j) for i in range(n) for j in range(m)]


def energy_all(W):
    """E(v,h) = -v^T W h over all (v,h) in {+-1}^(n+m), returned as (2^n, 2^m) array."""
    n, m = W.shape
    V = np.array(list(itertools.product([1, -1], repeat=n)))
    Hh = np.array(list(itertools.product([1, -1], repeat=m)))
    return -(V @ W @ Hh.T), V, Hh


def gauge_to_gs(W):
    E, V, Hh = energy_all(W)
    a, b = np.unravel_index(np.argmin(E), E.shape)
    return W * np.outer(V[a], Hh[b])


def plus_is_gs(W, tol=1e-9):
    E, _, _ = energy_all(W)
    return E[0, 0] <= E.min() + tol


def loop_atoms(n, m):
    idx = {e: k for k, e in enumerate(edges(n, m))}
    atoms = []
    for i1, i2 in itertools.combinations(range(n), 2):
        for j1, j2 in itertools.combinations(range(m), 2):
            cyc = [(i1, j1), (i1, j2), (i2, j1), (i2, j2)]
            for neg in cyc:
                a = np.zeros(n * m)
                for e in cyc:
                    a[idx[e]] = -1.0 if e == neg else 1.0
                atoms.append(a)
    return np.array(atoms).T  # (nm, K)


def in_loop_cone(W):
    """Feasibility of L x <= w, x >= 0 (statement (7))."""
    n, m = W.shape
    L = loop_atoms(n, m)
    w = W.reshape(-1)
    res = linprog(np.zeros(L.shape[1]), A_ub=L, b_ub=w, bounds=(0, None), method="highs")
    return res.status == 0


def frustration(W):
    """Weight fraction of negative entries in the ground-state gauge."""
    G = gauge_to_gs(W)
    return -G[G < 0].sum() / np.abs(G).sum()


def pentagonal_k44(M):
    """Lift of the K5 pentagonal facet sum b_i b_j x_ij <= 0, b = (1,1,1,-1,-1), to K_{4,4}.

    Contract v_k h_k (k = 0,1,2) with weight M; K5 vertices A,B,C = v0h0, v1h1, v2h2, D = v3, E = h3.
    Gauged weight on K5 pair (p,q) is -b_p b_q; parallel K_{4,4} edges get the weight on one copy.
    """
    W = np.zeros((4, 4))
    for k in range(3):
        W[k, k] = M
    W[0, 1] = W[0, 2] = W[1, 2] = -1.0  # A-B, A-C, B-C (same side, b_p b_q = +1)
    W[3, 3] = -1.0                       # D-E
    W[3, 0] = W[3, 1] = W[3, 2] = 1.0    # D-A, D-B, D-C
    W[0, 3] = W[1, 3] = W[2, 3] = 1.0    # A-E, B-E, C-E
    return W


def main():
    rng = np.random.default_rng(0)
    print("== pentagonal lift on K_{4,4}")
    for M in [1, 2, 3, 4, 6, 10]:
        W = pentagonal_k44(float(M))
        print(f"M={M:>2}: +1 is GS: {plus_is_gs(W)}, in loop cone: {in_loop_cone(W)}, "
              f"f = {frustration(W):.4f}" if plus_is_gs(W) else f"M={M:>2}: +1 not a GS")
    print("== random Gaussian RBMs, gauged to their ground state")
    for n, m, trials in [(2, 5, 200), (3, 3, 300), (3, 5, 200), (3, 6, 100), (4, 4, 300), (4, 5, 200), (5, 5, 100)]:
        bad = 0
        for _ in range(trials):
            W = gauge_to_gs(rng.standard_normal((n, m)))
            if not in_loop_cone(W):
                bad += 1
        print(f"{n}x{m}: {bad}/{trials} gauged instances NOT in the loop cone")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
