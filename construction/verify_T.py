"""End-to-end check of the reduction NAE-3SAT -> TILE (theorem T) on small formulas.

For each formula: build the instance, run the structural checks (hypotheses of the lemmas), then decide TILE with an
independent MILP over the raw couplings (ground-state sets recomputed from W), both unconstrained and with every
variable network pinned to each of the 2^n assignments.
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import build, structural_checks  # noqa: E402
from tiles import CORNERS, add, check_solution, tile_milp  # noqa: E402

FORMULAS = {
    "single": (3, [[(0, 0), (1, 0), (2, 0)]]),
    "sat3": (3, [[(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (2, 1)], [(0, 0), (1, 1), (2, 0)]]),
    "unsat4": (3, [[(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (2, 1)], [(0, 0), (1, 1), (2, 0)],
                   [(0, 1), (1, 0), (2, 0)]]),
}


def nae_ok(clauses, assign):
    return all(len({assign[v] ^ n for v, n in cl}) > 1 for cl in clauses)


def main(names):
    L = int(os.environ.get("TILE_L", "24"))
    h = L // 4; centers_all = [(h - h % 2, h - h % 2, L // 2), (3 * h - (3 * h) % 2, h - h % 2, L // 2), (h - h % 2, 3 * h - (3 * h) % 2, L // 2), (3 * h - (3 * h) % 2, 3 * h - (3 * h) % 2, L // 2)]
    only = os.environ.get("TILE_ONLY")  # "free" or an assignment index
    for name in names:
        n, clauses = FORMULAS[name]
        t0 = time.time()
        inst, Wall = build(L, clauses, centers_all[:len(clauses)])
        out, label = structural_checks(inst, Wall)
        print(f"[{name}] L={L} cells={len(Wall)} build {time.time()-t0:.1f}s structural: {out}", flush=True)
        # one representative vertex per variable network, and one background vertex
        rep = {}
        for (c, k), lab in label.items():
            if lab.startswith("x") and lab not in rep:
                rep[lab] = add(c, CORNERS[k], L)
        bvert = None
        for (c, k), lab in label.items():
            if lab == "B":
                bvert = add(c, CORNERS[k], L)
                break
        t0 = time.time()
        if only not in (None, "free"):
            idx = [int(x) for x in only.split(",")]
        else:
            idx = None
        if idx is not None:
            feas = None
            sat = any(nae_ok(clauses, a) for a in itertools.product([0, 1], repeat=n))
        else:
            feas, bits = tile_milp(L, Wall)
            sat = any(nae_ok(clauses, a) for a in itertools.product([0, 1], repeat=n))
            print(f"  TILE feasible={feas} (independent check of returned s: {feas and check_solution(L, Wall, bits)}),"
                  f" formula NAE-satisfiable={sat}  [{time.time()-t0:.1f}s]", flush=True)
            if only == "free":
                continue
        agree = 0
        tried = 0
        for ai, a in enumerate(itertools.product([0, 1], repeat=n)):
            if idx is not None and ai not in idx:
                continue
            tried += 1
            fix = {bvert: 0}
            for v in range(n):
                fix[rep[f"x{v}"]] = a[v]
            f, b = tile_milp(L, Wall, fix=fix)
            want = nae_ok(clauses, a)
            agree += (f == want)
            if f != want:
                print(f"  MISMATCH assignment {a}: TILE {f} vs NAE {want}")
        print(f"  pinned assignments {idx if idx is not None else 'all'} agreeing with NAE: {agree}/{tried}  [{time.time()-t0:.1f}s total]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["single"])
