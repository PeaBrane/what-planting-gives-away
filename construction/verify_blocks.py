"""End-to-end check of the explicit block layout (layout A) for theorem T.

For each formula: layout A (coarse wiring lemma checked), expansion with the oriented block templates and the
cluster template, translation into the torus; then
  (a) well-typedness of every cell (hypotheses of the network lemma, from raw couplings), class counts;
  (b) every network connected;
  (c) an independent exact TILE decision (lean_tile: parity union-find + MILP from raw couplings), unconstrained
      and with every variable network pinned to each assignment; compared with NAE-satisfiability.
Usage: verify_blocks.py NAME [NAME ...]   (names in FORMULAS, or rand:<n>:<m>:<seed>)
"""
import gc
import itertools
import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import lean_tile as LT  # noqa: E402
from layout import expand, layout_A, to_torus  # noqa: E402
from tiles import CORNERS, DIRS, DIR_TO_K  # noqa: E402

FORMULAS = {
    "single": (3, [[(0, 0), (1, 0), (2, 0)]]),
    "single_neg": (3, [[(0, 1), (1, 0), (2, 1)]]),
    "pair": (3, [[(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)]]),
    "sat3": (3, [[(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (2, 1)], [(0, 0), (1, 1), (2, 0)]]),
    "unsat4": (3, [[(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (2, 1)], [(0, 0), (1, 1), (2, 0)],
                   [(0, 1), (1, 0), (2, 0)]]),
}


def random_formula(n, m, seed):
    rng = random.Random(seed)
    cls = []
    for _ in range(m):
        vs = rng.sample(range(n), 3)
        cls.append([(v, rng.randint(0, 1)) for v in vs])
    return n, cls


def nae_ok(clauses, a):
    return all(len({a[v] ^ ng for v, ng in cl}) > 1 for cl in clauses)


def networks(L, netc):
    cells_of = {}
    for c, d in netc.items():
        for k, nt in d.items():
            cells_of.setdefault(nt, set()).add(c)
    conn = {}
    for nt, cells in cells_of.items():
        st = next(iter(cells))
        seen, dq = {st}, deque([st])
        while dq:
            c = dq.popleft()
            for k, lab in netc[c].items():
                if lab != nt:
                    continue
                n = tuple((x + a) % L for x, a in zip(c, DIRS[k]))
                assert netc.get(n, {}).get(DIR_TO_K[tuple(-a for a in DIRS[k])]) == nt, "label mismatch"
                if n not in seen:
                    seen.add(n)
                    dq.append(n)
        conn[nt] = len(seen) == len(cells)
    return cells_of, conn


def main(names):
    pin_cap = int(os.environ.get("PIN_CAP", "8"))
    for name in names:
        if name.startswith("rand:"):
            _, n, m, seed = name.split(":")
            n_vars, clauses = random_formula(int(n), int(m), int(seed))
        else:
            n_vars, clauses = FORMULAS[name]
        t0 = time.time()
        origins, edges, paths = layout_A(clauses)
        lk = expand(clauses, origins, edges, paths)
        L, Wc, netc, shift = to_torus(lk, int(os.environ.get("MARGIN", "1")))
        W_of = lambda c: Wc.get(c, B.W_BG)  # noqa: E731
        cells_W = ((c, W_of(c)) for c in LT.all_cells(L))
        counts, clause_rel = B.check_well_typed(cells_W, lambda c: netc.get(c, {}), L)
        cells_of, conn = networks(L, netc)
        heights = sorted({max(g[2] for g in p) for p in paths})
        print(f"[{name}] n={n_vars} m={len(clauses)} clauses={clauses} margin_blocks={os.environ.get('MARGIN', '1')}",
              flush=True)
        print(f"  layout A: {len(paths)} loop edges, coarse heights {heights}; L={L} N={L**3} cells={L**3//4};"
              f" well-typed: OK; classes {counts}; networks {len(conn)} all connected: {all(conn.values())}"
              f"  [{time.time()-t0:.1f}s]", flush=True)
        sat = any(nae_ok(clauses, a) for a in itertools.product([0, 1], repeat=n_vars))
        t0 = time.time()
        feas, bits = LT.decide(L, W_of, want_bits=True)
        okbits = feas and LT.check_bits(L, W_of, bits)
        print(f"  TILE planted={feas} (returned state re-checked cell by cell: {okbits}); formula NAE-satisfiable={sat};"
              f" agree={feas == sat}  [{time.time()-t0:.1f}s]", flush=True)
        if feas:
            # read off the assignment from the ground state and check it
            rep = {}
            for nt, cells in cells_of.items():
                c = next(iter(cells))
                k = next(k for k, lab in netc[c].items() if lab == nt)
                rep[nt] = tuple((x + d) % L for x, d in zip(c, CORNERS[k]))
            a = [bits[LT.vindex(L, rep[f"x{v}"])] if f"x{v}" in rep else 0 for v in range(n_vars)]
            print(f"  read-off assignment {a} NAE-satisfies the formula: {nae_ok(clauses, a)}", flush=True)
        else:
            rep = {}
            for nt, cells in cells_of.items():
                c = next(iter(cells))
                k = next(k for k, lab in netc[c].items() if lab == nt)
                rep[nt] = tuple((x + d) % L for x, d in zip(c, CORNERS[k]))
        bits = None
        gc.collect()
        t0 = time.time()
        assigns = list(itertools.product([0, 1], repeat=n_vars))
        if len(assigns) > pin_cap:
            rng = random.Random(1)
            assigns = rng.sample(assigns, pin_cap)
        agree = 0
        for a in assigns:
            fix = {rep[f"x{v}"]: a[v] for v in range(n_vars) if f"x{v}" in rep}
            f, _ = LT.decide(L, W_of, fix=fix)
            gc.collect()
            want = nae_ok(clauses, a)
            agree += f == want
            if f != want:
                print(f"  MISMATCH {a}: TILE {f} NAE {want}", flush=True)
        import resource
        print(f"  pinned assignments agreeing with NAE: {agree}/{len(assigns)}  [{time.time()-t0:.1f}s]"
              f"  peak RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["single"])
