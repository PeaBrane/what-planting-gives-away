"""End-to-end check of the G46 (F6 u F42 only) construction on the polarity-aware block layout.
Usage: verify46.py NAME [...]  (names as in verify_blocks.py)."""
import gc
import itertools
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_tile as LT  # noqa: E402
import layout46 as G  # noqa: E402
from tiles import CORNERS  # noqa: E402
from verify_blocks import FORMULAS, nae_ok, random_formula  # noqa: E402


def networks_no_clause(L, netc, role):
    """Each network's cells (wires, fan-outs) connected through shared labelled corners, never passing through a
    clause cell (so the clause relation is not used to connect the two corners of a clause domino)."""
    from collections import deque
    from tiles import DIRS
    cells_of = {}
    for c, d in netc.items():
        if role[c].startswith("clause"):
            continue
        for k, nt in d.items():
            cells_of.setdefault(nt, set()).add(c)
    conn = {}
    for nt, cells in cells_of.items():
        st = next(iter(cells))
        seen, dq = {st}, deque([st])
        while dq:
            c = dq.popleft()
            for k, lab in netc[c].items():
                n = tuple((x + a) % L for x, a in zip(c, DIRS[k]))
                if lab == nt and n in cells and n not in seen:
                    seen.add(n)
                    dq.append(n)
        conn[nt] = len(seen) == len(cells)
    return cells_of, conn


def main(names):
    pin_cap = int(os.environ.get("PIN_CAP", "8"))
    margin = int(os.environ.get("MARGIN", "1"))
    for name in names:
        if name.startswith("rand:"):
            _, n, m, seed = name.split(":")
            n_vars, clauses = random_formula(int(n), int(m), int(seed))
        else:
            n_vars, clauses = FORMULAS[name]
        t0 = time.time()
        origins, edges, paths = G.layout_A46(clauses)
        lk, negs = G.expand46(clauses, origins, edges, paths)
        L, netc, role = G.to_torus46(lk, negs, margin)
        Wc, info, uf, root = G.assign_couplings46(L, netc, role, negs)
        counts = G.check_lemma46(L, netc, role, Wc, uf, root)
        # networks connected through wire and fan-out cells only (clause cells excluded)
        cells_of, conn = networks_no_clause(L, netc, role)
        heights = sorted({max(g[2] for g in p) for p in paths})
        print(f"[{name}] G46 n={n_vars} m={len(clauses)} clauses={clauses} margin_blocks={margin}", flush=True)
        print(f"  layout A: {len(paths)} loop edges, heights {heights}; L={L} cells={L**3//4}; classes {counts};"
              f" lemma hypotheses OK; anchor {info}; networks {len(conn)} connected w/o clause cells:"
              f" {all(conn.values())}  [{time.time()-t0:.1f}s]", flush=True)
        W_of = lambda u: Wc.get(u) or G.bg_coupling(u)  # noqa: E731
        sat = any(nae_ok(clauses, a) for a in itertools.product([0, 1], repeat=n_vars))
        t0 = time.time()
        feas, bits = LT.decide(L, W_of, want_bits=True)
        okb = feas and LT.check_bits(L, W_of, bits)
        print(f"  TILE planted={feas} (state re-checked: {okb}); NAE-sat={sat}; agree={feas == sat}"
              f"  [{time.time()-t0:.1f}s]", flush=True)
        rep = {}
        for nt, cells in cells_of.items():
            c = next(iter(cells))
            k = next(k for k, lab in netc[c].items() if lab == nt)
            rep[nt] = tuple((x + d) % L for x, d in zip(c, CORNERS[k]))
        if feas:
            a = [bits[LT.vindex(L, rep[f"x{v}"])] if f"x{v}" in rep else 0 for v in range(n_vars)]
            print(f"  read-off assignment {a} NAE-satisfies: {nae_ok(clauses, a)}", flush=True)
        bits = None
        gc.collect()
        assigns = list(itertools.product([0, 1], repeat=n_vars))
        if len(assigns) > pin_cap:
            assigns = random.Random(1).sample(assigns, pin_cap)
        agree = 0
        t0 = time.time()
        for a in assigns:
            fix = {rep[f"x{v}"]: a[v] for v in range(n_vars) if f"x{v}" in rep}
            f, _ = LT.decide(L, W_of, fix=fix)
            gc.collect()
            want = nae_ok(clauses, a)
            agree += f == want
            if f != want:
                print(f"  MISMATCH {a}: TILE {f} NAE {want}", flush=True)
        print(f"  pinned agree {agree}/{len(assigns)}  [{time.time()-t0:.1f}s]  peak RSS"
              f" {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["single"])
