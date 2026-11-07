"""Cycle (metric) relaxation of the Ising ground-state problem on frustrated-loop instances.

Claim under test: if H = sum of frustrated loops each minimised by s*, the cycle LP value equals
H(s*) exactly; if every loop has a strictly unique ground state, the LP optimum is s* itself.
Random +-J instances on the same lattice are a comparison (LP is only a lower bound there).
"""
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def cubic_torus(L):
    idx = lambda x, y, z: (x % L) * L * L + (y % L) * L + (z % L)
    E = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                u = idx(x, y, z)
                E += [(u, idx(x + 1, y, z)), (u, idx(x, y + 1, z)), (u, idx(x, y, z + 1))]
    nbr = [[] for _ in range(L ** 3)]
    for k, (u, v) in enumerate(E):
        nbr[u].append((v, k))
        nbr[v].append((u, k))
    return L ** 3, E, nbr


def random_loop(rng, N, nbr):
    """Hen et al.: random walk until it revisits a vertex; keep the cycle (edge ids)."""
    v = int(rng.integers(N))
    pos = {v: 0}
    path_v, path_e = [v], []
    prev_e = -1
    while True:
        choices = [(w, k) for (w, k) in nbr[v] if k != prev_e]
        w, k = choices[rng.integers(len(choices))]
        path_e.append(k)
        if w in pos:
            return path_e[pos[w]:]
        pos[w] = len(path_v)
        path_v.append(w)
        v, prev_e = w, k


def loop_instance(rng, N, E, nbr, n_loops, neg=1.0):
    """Couplings J (H = -sum J s_i s_j) gauged so s* = +1: each loop is +1 on all edges but one, -neg there."""
    J = np.zeros(len(E))
    E0 = 0.0
    for _ in range(n_loops):
        cyc = random_loop(rng, N, nbr)
        bad = cyc[rng.integers(len(cyc))]
        for k in cyc:
            J[k] += 1.0
        J[bad] -= 1.0 + neg
        E0 += -(len(cyc) - 1) + neg  # loop energy at s* = +1
    return J, E0


def cycle_lp(N, E, J, max_rounds=200, tol=1e-7, max_new=50, partial=False):
    """min sum_e J_e x_e over the metric polytope, by cutting planes; H = -sum J + 2 * LP."""
    m = len(E)
    rows, rhs = [], []
    u = np.array([e[0] for e in E])
    v = np.array([e[1] for e in E])
    x = None
    for _ in range(max_rounds):
        A = np.array(rows) if rows else None
        res = linprog(J, A_ub=A, b_ub=np.array(rhs) if rows else None, bounds=(0, 1), method="highs")
        x = np.clip(res.x, 0.0, 1.0)
        # doubled graph: node (w, p) = w + p * N; even arcs weight x, odd arcs weight 1 - x
        src = np.concatenate([u, u + N, u, u + N])
        dst = np.concatenate([v, v + N, v + N, v])
        wts = np.concatenate([x, x, 1 - x, 1 - x]) + 1e-12
        eid = np.concatenate([np.arange(m)] * 4)
        odd = np.concatenate([np.zeros(2 * m, bool), np.ones(2 * m, bool)])
        G = coo_matrix((wts, (src, dst)), shape=(2 * N, 2 * N)).tocsr()
        G = G + G.T
        arc = {}
        for a, b, k, o in zip(src, dst, eid, odd):
            arc[(a, b)] = (k, o)
            arc[(b, a)] = (k, o)
        dist, pred = dijkstra(G, indices=np.arange(N), return_predecessors=True)
        new = 0
        seen = set()
        for s in np.argsort(dist[np.arange(N), np.arange(N) + N]):
            if dist[s, s + N] >= 1 - tol or new >= max_new:
                break
            walk, t = [], s + N
            while t != s:
                p = pred[s, t]
                walk.append(arc[(p, t)])
                t = p
            F = [k for k, o in walk if o]
            C = [k for k, o in walk if not o]
            key = (tuple(sorted(F)), tuple(sorted(C)))
            if key in seen or len(set(F) | set(C)) != len(walk):
                continue
            seen.add(key)
            row = np.zeros(m)
            row[F] += 1.0
            row[C] -= 1.0
            rows.append(row)
            rhs.append(len(F) - 1.0)
            new += 1
        if new == 0:
            return -J.sum() + 2 * res.fun, x, len(rows)
    return ((-J.sum() + 2 * res.fun) if partial else None), x, len(rows)


def metropolis_best(rng, N, E, J, sweeps=3000):
    nbr = [[] for _ in range(N)]
    for k, (a, b) in enumerate(E):
        nbr[a].append((b, J[k]))
        nbr[b].append((a, J[k]))
    s = rng.choice([-1, 1], N)
    energy = lambda s: -sum(J[k] * s[a] * s[b] for k, (a, b) in enumerate(E))
    best = energy(s)
    cur = best
    for t in range(sweeps):
        beta = 0.1 + 3.0 * t / sweeps
        for i in rng.permutation(N):
            dE = 2 * s[i] * sum(Jk * s[j] for j, Jk in nbr[i])
            if dE <= 0 or rng.random() < np.exp(-beta * dE):
                s[i] = -s[i]
                cur += dE
        best = min(best, cur)
    return best


def main():
    rng = np.random.default_rng(1)
    L = 4
    N, E, nbr = cubic_torus(L)
    eu = np.array([e[0] for e in E]); ev = np.array([e[1] for e in E])
    print(f"3D torus L={L}: N={N}, |E|={len(E)}; instances are randomly gauged before solving")
    for label, neg in [("+-1 loops (Hen)", 1.0), ("strict loops, neg=0.5", 0.5)]:
        for alpha in [0.1, 0.3, 1.0, 2.0]:
            tight = integral = planted = 0
            trials = 4
            for _ in range(trials):
                J, E0 = loop_instance(rng, N, E, nbr, int(alpha * N), neg)
                t = rng.choice([-1, 1], N)
                Jg = J * t[eu] * t[ev]
                xstar = (t[eu] != t[ev]).astype(float)
                lp, x, ncuts = cycle_lp(N, E, Jg)
                tight += lp is not None and abs(lp - E0) < 1e-5
                if np.all(np.minimum(x, 1 - x) < 1e-6):
                    integral += 1
                    sup = np.abs(J) > 1e-12
                    planted += np.all(np.abs(x[sup] - xstar[sup]) < 1e-6)
            print(f"{label:24s} alpha={alpha:3.1f}: LP == E0 {tight}/{trials}, integral LP vertex {integral}/{trials}, = planted cut {planted}/{trials}", flush=True)
    print("random +-J (no planting):", flush=True)
    for _ in range(3):
        J = rng.choice([-1.0, 1.0], len(E))
        lp, x, ncuts = cycle_lp(N, E, J, max_rounds=60)
        print(f"  LP lower bound = {lp}, integral = {np.all(np.minimum(x, 1 - x) < 1e-6)}, cuts = {ncuts}", flush=True)


if __name__ == "__main__":
    main()
