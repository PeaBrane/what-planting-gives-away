"""global anchoring check (construction notes Sec. 8.5) on full G46 layout-A instances, plus optional end-to-end TILE.
Usage: TILE_S=12 MARGIN=m [E2E=1] anchor46.py NAME [...]   (names as in verify_blocks.FORMULAS or rand:n:m:seed)
Checks, on the torus instance actually built:
  (a) check_lemma46: every cell's class/ground states, and propagated anchoring from the largest lock component
      pins every F6 wire cell (all cluster regions included);
  (b) foam-giant containment: for EVERY block not in a cluster region, the corners of its foam cells that lie in
      the periodic foam giant (offsets taken from the 4x4x4 periodic torus, g46_local2.L1) all lie in ONE lock
      component K of the instance, and K is the anchor component A0 used in (a);
  (c) [E2E=1] TILE decided by the exact decider; planted <=> NAE-sat; read-off; pinned assignments."""
import gc
import itertools
import math
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
import lean_tile as LT  # noqa: E402
from g46_local2 import L1, corner as pcorner, foam  # noqa: E402
from tiles import CORNERS  # noqa: E402
from verify46 import networks_no_clause  # noqa: E402
from verify_blocks import FORMULAS, nae_ok, random_formula  # noqa: E402

S, H = B.S, B.H
assert S == 12


def giant_offsets():
    """Offsets (relative to the block centre S*g) of foam-cell corners of one block lying in the periodic giant."""
    _, (P, cells, uf, big) = L1()
    assert len(big) == 1
    g0 = big[0]
    C = (S, S, S)
    offs = set()
    for u in cells:
        if not all(C[j] - H <= u[j] < C[j] + H for j in range(3)):
            continue
        for k in range(8):
            if uf.find(pcorner(u, k, P)) == g0:
                offs.add(tuple(u[j] + CORNERS[k][j] - C[j] for j in range(3)))
    return sorted(offs)


def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def run(name, margin, e2e, offs):
    if name.startswith("rand:"):
        _, n, m, seed = name.split(":")
        n_vars, clauses = random_formula(int(n), int(m), int(seed))
    else:
        n_vars, clauses = FORMULAS[name]
    t0 = time.time()
    origins, edges, paths = G.layout_A46(clauses)
    lk, negs = G.expand46(clauses, origins, edges, paths)
    lo = [min(c[j] for c in lk.used) for j in range(3)]
    kshift = tuple(margin - math.floor(l / S) for l in lo)
    L, netc, role = G.to_torus46(lk, negs, margin)
    nb = L // S
    Wc, info, uf, root = G.assign_couplings46(L, netc, role, negs)
    counts = G.check_lemma46(L, netc, role, Wc, uf, root)
    heights = sorted({max(g[2] for g in p) for p in paths})
    print(f"[{name}] n={n_vars} m={len(clauses)} clauses={clauses} origins={origins} margin={margin}", flush=True)
    print(f"  L={L} ({nb} blocks/axis) heights={heights}; classes {counts}; lemma 8.1 hypotheses OK (all F6 wire"
          f" cells pinned by propagated anchoring); anchor {info}  [{time.time()-t0:.1f}s, RSS {mem():.0f} MB]",
          flush=True)
    # (b) foam-giant containment
    t0 = time.time()
    region = set()
    for O in origins:
        for a, b, z in itertools.product(range(6), range(4), (-1, 0)):
            region.add(tuple((O[j] + d + kshift[j]) % nb for j, d in enumerate((a, b, z))))
    cnt = {}
    for u in LT.all_cells(L):
        lab = netc.get(u, {})
        for k in range(8):
            if k not in lab:
                r, _ = uf.find(LT.corner_index(L, u, k))
                cnt[r] = cnt.get(r, 0) + 1
    A0 = max(cnt, key=cnt.get)
    roots = {}
    nblk = 0
    for g in itertools.product(range(nb), repeat=3):
        if g in region:
            continue
        nblk += 1
        for d in offs:
            v = tuple(S * g[j] + d[j] for j in range(3))
            r, _ = uf.find(LT.vindex(L, v))
            roots[r] = roots.get(r, 0) + 1
    print(f"  (b) foam-giant containment: {nblk} non-region blocks x {len(offs)} giant refs; region blocks"
          f" {len(region)}; distinct lock components among refs: {len(roots)}; equals A0: {set(roots) == {A0}}"
          f"  [{time.time()-t0:.1f}s]", flush=True)
    assert set(roots) == {A0}, "global anchoring FAILS"
    if not e2e:
        return
    cells_of, conn = networks_no_clause(L, netc, role)
    print(f"  networks {len(conn)} connected w/o clause cells: {all(conn.values())}", flush=True)
    assert all(conn.values())
    W_of = lambda u: Wc.get(u) or G.bg_coupling(u)  # noqa: E731
    sat = any(nae_ok(clauses, a) for a in itertools.product([0, 1], repeat=n_vars))
    t0 = time.time()
    tl = float(os.environ.get("TL", "900"))
    feas, bits = LT.decide(L, W_of, want_bits=True, time_limit=tl)
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
    cap = int(os.environ.get("PIN_CAP", "8"))
    if len(assigns) > cap:
        assigns = random.Random(1).sample(assigns, cap)
    agree = 0
    t0 = time.time()
    for a in assigns:
        fix = {rep[f"x{v}"]: a[v] for v in range(n_vars) if f"x{v}" in rep}
        f, _ = LT.decide(L, W_of, fix=fix, time_limit=tl)
        gc.collect()
        want = nae_ok(clauses, a)
        agree += f == want
        if f != want:
            print(f"  MISMATCH {a}: TILE {f} NAE {want}", flush=True)
    print(f"  pinned agree {agree}/{len(assigns)}  [{time.time()-t0:.1f}s]  peak RSS {mem():.0f} MB", flush=True)


if __name__ == "__main__":
    margin = int(os.environ.get("MARGIN", "3"))
    e2e = os.environ.get("E2E", "0") == "1"
    t0 = time.time()
    offs = giant_offsets()
    print(f"periodic foam giant: {len(offs)} distinct corner offsets per block  [{time.time()-t0:.1f}s]", flush=True)
    for nm in sys.argv[1:]:
        run(nm, margin, e2e, offs)
