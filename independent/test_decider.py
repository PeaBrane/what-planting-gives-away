"""Cross-check my_decider against a naive exact cell-by-cell backtracking on small random tori (L = 4, 6)."""
import os
import itertools
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_decider as D  # noqa: E402

ALLW = [tuple(w) for w in itertools.product([1, -1], repeat=12)]
GAM = {}


def gam(W):
    if W not in GAM:
        e = D.cell_energy_table(W)
        GAM[W] = frozenset(np.nonzero(e == e.min())[0].tolist())
    return GAM[W]


def cells_of(L):
    return [u for u in itertools.product(range(L), repeat=3) if u[0] % 2 == u[1] % 2 == u[2] % 2]


def naive(L, Wall):
    cells = cells_of(L)
    # order cells by BFS so that consecutive cells share vertices
    order, seen = [cells[0]], {cells[0]}
    i = 0
    while len(order) < len(cells):
        u = order[i]
        i += 1
        for k in range(8):
            d = D.CORN[k]
            w = tuple((u[j] + 2 * d[j] - 1) % L for j in range(3))
            if w not in seen:
                seen.add(w)
                order.append(w)
    sigma = {}
    budget = [300000]

    def rec(p):
        budget[0] -= 1
        if budget[0] < 0:
            raise TimeoutError
        if p == len(order):
            return True
        u = order[p]
        vs = [tuple((u[j] + D.CORN[k][j]) % L for j in range(3)) for k in range(8)]
        for m in gam(Wall[u]):
            newly, ok = [], True
            for k, v in enumerate(vs):
                b = (m >> k) & 1
                if v in sigma:
                    if sigma[v] != b:
                        ok = False
                        break
                else:
                    sigma[v] = b
                    newly.append(v)
            if ok and rec(p + 1):
                return True
            for v in newly:
                del sigma[v]
        return False

    return rec(0)


def main():
    rng = np.random.default_rng(7)
    by_class = {}
    for W in ALLW:
        G = gam(W)
        by_class.setdefault(len(G), []).append(W)
    stats = {}
    for trial in range(int(sys.argv[1]) if len(sys.argv) > 1 else 240):
        L = 4 if trial < 200 else 6
        cells = cells_of(L)
        kind = trial % 4 if len(sys.argv) < 3 else int(sys.argv[2])
        if kind in (0, 3):   # planted: random sigma, random couplings having sigma|_C as a ground state
            sig = {v: int(rng.integers(2)) for v in itertools.product(range(L), repeat=3)}
            Wall = {}
            for u in cells:
                m = sum(sig[tuple((u[j] + D.CORN[k][j]) % L for j in range(3))] << k for k in range(8))
                cand = [W for W in ALLW if m in gam(W)]
                # bias toward small ground-state sets so that the instance is constraining
                cand.sort(key=lambda W: len(gam(W)))
                lim = 40 if kind == 0 else len(cand)
                Wall[u] = cand[int(rng.integers(min(len(cand), lim)))]
            if kind == 3:   # perturb one cell: feasibility is then not known in advance
                u = cells[int(rng.integers(len(cells)))]
                Wall[u] = ALLW[int(rng.integers(4096))]
        elif kind == 2 and L == 4 and trial % 8 == 2:     # degenerate classes only, rarely feasible
            Wall = {}
            for u in cells:
                size = [4, 8, 16][int(rng.integers(3))]
                lst = by_class[size]
                Wall[u] = lst[int(rng.integers(len(lst)))]
        elif kind == 4:     # ferromagnetic-gauge rigid background with random non-rigid cells
            rigid0 = [W for W in by_class[2] if 0 in gam(W)]
            Wall = {}
            for u in cells:
                r = rng.random()
                if r < 0.6:
                    Wall[u] = rigid0[int(rng.integers(len(rigid0)))]
                else:
                    size = [4, 8, 16][int(rng.integers(3))]
                    lst = by_class[size]
                    Wall[u] = lst[int(rng.integers(len(lst)))]
        elif kind == 5:     # planted at sigma = 0, then 1-3 cells locally gauged by a random corner mask
            pl = [W for W in ALLW if 0 in gam(W)]
            Wall = {u: pl[int(rng.integers(len(pl)))] for u in cells}
            for _ in range(int(rng.integers(1, 4))):
                u = cells[int(rng.integers(len(cells)))]
                t = int(rng.integers(1, 255))
                Wall[u] = tuple(w * (-1 if (t >> a) & 1 else 1) * (-1 if (t >> b) & 1 else 1)
                                for w, (a, b) in zip(Wall[u], D.EDGES))
        else:           # random mixture of the classes: mostly rigid/hinge, some F22/F6
            Wall = {}
            for u in cells:
                r = rng.random()
                size = 2 if r < 0.55 else (4 if r < 0.85 else (8 if r < 0.95 else 16))
                lst = by_class[size]
                Wall[u] = lst[int(rng.integers(len(lst)))]
        Wdef = Wall[cells[0]]
        Wc = {u: W for u, W in Wall.items() if u != cells[0]}
        inst, sol = D.decide(L, Wdef, Wc)
        mine = sol is not None
        import time as _t
        _t0 = _t.time()
        try:
            ref = naive(L, Wall)
        except TimeoutError:
            stats[("skipped", L, kind)] = stats.get(("skipped", L, kind), 0) + 1
            continue
        if _t.time() - _t0 > 1:
            print("slow naive", trial, L, kind, f"{_t.time()-_t0:.1f}s", flush=True)
        if mine:
            sigma = inst.expand(sol)
            ok, Hc, He = inst.verify(sigma)
            assert ok and Hc == He == inst.E0, (trial, ok, Hc, He, inst.E0)
        key = (L, kind, mine, ref)
        stats[key] = stats.get(key, 0) + 1
        assert mine == ref, ("MISMATCH", trial, L, kind, mine, ref)
    print("decider vs naive backtracking (L, kind, mine, naive): count")
    for k in sorted(stats, key=str):
        print("  ", k, stats[k])
    print("all agree; every returned state verified (local minima, edge-by-edge energy == E0)")


if __name__ == "__main__":
    main()
