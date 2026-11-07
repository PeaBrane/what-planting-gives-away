"""own (W1), (W3), (W4) checks from raw couplings.  The construction data (cluster template cells and
links, the 30 templates) come from the construction builders, since they are the object under test; every lock, component,
interface set and pinning step is recomputed here (own_cube, own_geom).
Run: TILE_S=12 python -I own_windows.py [W0 W1 W3 W4 W2 W4strict]"""
import itertools
import os
import resource
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_cube as O  # noqa: E402
import own_geom as Gm  # noqa: E402
from own_geom import DIR_K, DIRS, H, S, SLAB, add, sub  # noqa: E402

os.environ.setdefault("TILE_S", "12")
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "construction"))


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ------------------------------------------------------------------ periodic foam giant (L1) and its offsets
def periodic_giant(nb=4):
    P = S * nb

    def vid(v):
        return ((v[0] % P) * P + v[1] % P) * P + v[2] % P

    src, dst, cells = [], [], []
    for g in itertools.product(range(nb), repeat=3):
        for u in Gm.block_foam(g):
            u = tuple(x % P for x in u)
            cells.append(u)
            for j, k, p in Gm.bg_locks(sum(u) % 3):
                assert p == 0
                src.append(vid(add(u, O.CORN[j])))
                dst.append(vid(add(u, O.CORN[k])))
    n = P ** 3
    A = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    inc = {}
    for u in cells:
        for k in range(8):
            r = int(lab[vid(add(u, O.CORN[k]))])
            inc[r] = inc.get(r, 0) + 1
    big = [r for r, c in inc.items() if c > 64]
    assert len(big) == 1
    gi = big[0]
    C = (S, S, S)
    offs = {}  # corner offset -> list of (cell offset) it comes from
    for g in [(1, 1, 1)]:
        for u in Gm.block_foam(g):
            for k in range(8):
                v = add(u, O.CORN[k])
                if int(lab[vid(v)]) == gi:
                    offs.setdefault(sub(v, C), set()).add(sub(u, C))
    sizes = sorted(inc.values(), reverse=True)
    return dict(foam_cells=len(cells), components=len(inc), giant_incidences=inc[gi],
                largest_other=max([c for r, c in inc.items() if r != gi], default=0), sizes_top=sizes[:3]), offs


def iface_refs(b, b2, offs, strict=False):
    """I(b, b2): giant refs in the two facing shell layers.  Code definition: corner offset d_j >= H-1 (lower block)
    or <= 1-H (upper block).  strict: the ref must be a corner of a foam cell lying in the facing layer itself."""
    j = next(i for i in range(3) if b[i] != b2[i])
    lo, hi = (b, b2) if b[j] < b2[j] else (b2, b)
    out = []
    for d, srcs in offs.items():
        if d[j] >= H - 1 and (not strict or any(c[j] == H - 1 for c in srcs)):
            out.append(add(tuple(S * x for x in lo), d))
        if d[j] <= 1 - H and (not strict or any(c[j] == -H for c in srcs)):
            out.append(add(tuple(S * x for x in hi), d))
    return out


def robust(g, g2):
    if g[2] in SLAB or g2[2] in SLAB:
        return False
    return g[2] != g2[2] or g[2] not in (-2, 1)


def nbrs(g):
    for j in range(3):
        for s in (1, -1):
            yield tuple(g[i] + (s if i == j else 0) for i in range(3))


def window_box(blocks):
    lo = [min(S * g[j] for g in blocks) - H - 1 for j in range(3)]
    hi = [max(S * g[j] for g in blocks) + H + 1 for j in range(3)]
    return Gm.Box(lo, hi)


def W1(offs, strict=False):
    res = {}
    for zb in (-4, -3, -2, 1, 2, 3):
        b = (0, 0, zb)
        blocks = [add(b, d) for d in itertools.product(range(-2, 3), repeat=3)]
        pres = [g for g in blocks if g[2] not in SLAB]
        lg = Gm.LockGraph(window_box(blocks))
        for g in pres:
            lg.add_bg(Gm.block_foam(g))
        lg.solve()
        refs = [v for b2 in nbrs(b) if robust(b, b2) for v in iface_refs(b, b2, offs, strict)]
        res[zb] = (len({lg.comp(v) for v in refs}), len(refs))
    return dict(comps_refs=res, failing={k: v for k, v in res.items() if v[0] != 1})


def present_robust_refs(pres, offs):
    P = set(pres)
    out = []
    for g in P:
        for g2 in nbrs(g):
            if g2 in P and g < g2 and robust(g, g2):
                out += iface_refs(g, g2, offs)
    return out


def propagate(lg, anch, f6, links):
    """f6: {cell: D mask}; links: {cell: set of link corners}.  Least fixpoint of propagated anchoring."""
    pending = dict(f6)
    rounds = 0
    while pending:
        rounds += 1
        newly = []
        for c, D in pending.items():
            P = 0
            for k in range(8):
                if k not in links[c] and lg.comp(add(c, O.CORN[k])) in anch:
                    P |= 1 << k
            if O.pinnable(D, P):
                newly.append(c)
        if not newly:
            break
        for c in newly:
            del pending[c]
            for k in range(8):
                if k not in links[c]:
                    anch.add(lg.comp(add(c, O.CORN[k])))
    return sorted(pending), rounds


def template_links(cells):
    """Links of a template wire from its own cell sequence; the two end links continue the canonical crossing
    wire into the neighbouring block (the outer cell's across-face neighbour, recomputed here)."""
    links = {}
    n = len(cells)
    for t, c in enumerate(cells):
        ks = set()
        if t > 0:
            ks.add(DIR_K[sub(cells[t - 1], c)])
        if t < n - 1:
            ks.add(DIR_K[sub(cells[t + 1], c)])
        links[c] = ks
    return links


def end_link(c, block_g, tau):
    """Across-face partner of an outer crossing cell c of block g, for travel tau e_i: the outer cell of the
    neighbouring block's facing face on the same canonical wire (Lemma 4.4); asserted adjacent."""
    C = tuple(S * x for x in block_g)
    d = sub(c, C)
    i = next(j for j in range(3) if d[j] in (-H, H - 1))
    sg = 1 if d[i] == H - 1 else -1
    assert c == Gm.outer_cell(C, i, sg, tau), ("not the outer cell for this travel", c, block_g, tau)
    g2 = tuple(block_g[j] + (sg if j == i else 0) for j in range(3))
    x = Gm.outer_cell(tuple(S * y for y in g2), i, -sg, tau)
    assert all(abs(a) == 1 for a in sub(x, c)), (c, block_g, tau, x)
    return x


def W3(offs, zs=(1, 2, 3)):
    import templates46 as T6
    T = T6.templates46()
    fails, n = {}, 0
    for zb in zs:
        b = (0, 0, zb)
        C = tuple(S * x for x in b)
        blocks = [add(b, d) for d in itertools.product(range(-2, 3), repeat=3)]
        pres = [g for g in blocks if g[2] not in SLAB]
        anch_refs = present_robust_refs(pres, offs)
        for (f1, f2), tc in sorted(T.items()):
            n += 1
            cells = [add(C, c) for c in tc]
            links = template_links(cells)
            for c, tau in ((cells[0], -f1[1]), (cells[-1], f2[1])):
                x = end_link(c, b, tau)
                assert x not in cells
                links[c].add(DIR_K[sub(x, c)])
            assert all(len(v) == 2 and O.mask(v) in O.DOMINOES for v in links.values()), (f1, f2)
            lg = Gm.LockGraph(window_box(blocks))
            for g in pres:
                lg.add_bg(Gm.block_foam(g))
            lg.add_bg(Gm.block_interior(b, cells))
            lg.solve()
            anch = {lg.comp(v) for v in anch_refs}
            unp, _ = propagate(lg, anch, {c: O.mask(links[c]) for c in cells}, links)
            if unp:
                fails[zb, f1, f2] = len(unp)
    return dict(checks=n, failing=fails)


def cluster_data():
    import blocks as B
    import layout46 as G
    lk, assign, wires, u = G.cluster_template46()
    R = B.cluster_region((0, 0, 0))
    links = {c: {DIR_K[a] for a in lk.net[c]} for c in lk.used}
    # symmetry of links
    for c in lk.used:
        for k in links[c]:
            c2 = add(c, DIRS[k])
            assert c2 in lk.used and (7 - k) in links[c2], ("asymmetric link", c, k)
    roles = dict(lk.role)
    return lk, assign, wires, R, links, roles


def W4(offs, margin=2, with_sockets=True, start="robust"):
    import templates46 as T6
    lk, assign, wires, R, links, roles = cluster_data()
    T = T6.templates46()
    used = set(lk.used)
    outside = [c for c in used if not all(lo <= c[j] < hi for j, (lo, hi) in enumerate(R))]
    blocks = list(itertools.product(range(-margin, 6 + margin), range(-margin, 4 + margin), range(-1 - margin, 4)))
    pres = [g for g in blocks if g[2] not in SLAB]
    lg = Gm.LockGraph(window_box(blocks))
    for g in pres:
        lg.add_bg(Gm.block_foam(g))
    region_bg = [c for c in itertools.product(*[range(lo, hi) for lo, hi in R]) if Gm.iscell(c) and c not in used]
    lg.add_bg(region_bg)
    f6, fan = {}, []
    for c in used:
        r = roles[c]
        if r.startswith("clause"):
            assert len(links[c]) == 8
            continue
        if len(links[c]) == 4:
            fan.append(c)
            assert r.startswith("side")
        else:
            f6[c] = None
    import tiles as TF
    for c in fan:
        m = O.mask(links[c])
        assert frozenset(links[c]) in O.FACE_SETS
        W = TF.coupling_with_gs({0, m, 255 ^ m, 255}, "F42")  # instance rule for fan-outs
        lg.add_cell(c, Gm.locks_of(W))
    n_sock = 0
    socket_cells = []
    if with_sockets:
        for (i, pol), (a, bb) in assign.items():
            g = (2 * a, 2 * bb, 1)
            f1, f2 = ((2, -1), (2, 1)) if pol == 1 else ((2, 1), (2, -1))
            C = tuple(S * x for x in g)
            cells = [add(C, c) for c in T[f1, f2]]
            tl = template_links(cells)
            bot = cells[0] if pol == 1 else cells[-1]
            top = cells[-1] if pol == 1 else cells[0]
            w = wires[i, pol][-1]
            assert w == end_link(bot, g, pol), ("socket template does not meet the cluster wire", g)
            tl[bot].add(DIR_K[sub(w, bot)])
            links[w] = set(links[w]) | {DIR_K[sub(bot, w)]}
            x = end_link(top, g, pol)
            assert x not in cells
            tl[top].add(DIR_K[sub(x, top)])
            for c, ks in tl.items():
                links[c] = ks
                f6[c] = None
                socket_cells.append(c)
            lg.add_bg(Gm.block_interior(g, cells))
            n_sock += 1
    else:
        for (i, pol), w in wires.items():
            o = w[-1]
            g = tuple((o[j] + H) // S for j in range(3))
            x = end_link(o, g, pol)
            links[o] = set(links[o]) | {DIR_K[sub(x, o)]}
    for c in f6:
        assert len(links[c]) == 2 and O.mask(links[c]) in O.DOMINOES, (c, links[c])
        f6[c] = O.mask(links[c])
    lg.solve()
    if start == "robust":
        anch = {lg.comp(v) for v in present_robust_refs(pres, offs)}
    else:  # weaker start: robust refs of blocks at z >= 2 only (above the region) -- stress variant
        anch = {lg.comp(v) for v in present_robust_refs([g for g in pres if g[2] >= 2], offs)}
    unp, rounds = propagate(lg, anch, f6, links)
    model = dict(bg_cells=None, f6={c: frozenset(links[c]) for c in f6}, fan={c: frozenset(links[c]) for c in fan})
    return dict(cluster_cells_outside_region=len(outside), region_bg=len(region_bg), fanouts=len(fan),
                sockets=n_sock, f6_cells=len(f6), socket_f6=len(socket_cells), anchored_start=len(anch),
                unpinned=len(unp), rounds=rounds, example=unp[:3]), model


if __name__ == "__main__":
    which = sys.argv[1:] or ["W0", "W1", "W3", "W4"]
    t0 = time.time()
    info, offs = periodic_giant()
    print(f"(L1 own) {info}; giant offsets per block: {len(offs)}  [{time.time()-t0:.1f}s RSS {rss():.0f}MB]",
          flush=True)
    if "W0" in which:
        from anchor46 import giant_offsets
        fo = set(giant_offsets())
        print(f"  construction giant_offsets: {len(fo)}; equal to own: {fo == set(offs)}", flush=True)
    for nm in which:
        t1 = time.time()
        if nm == "W1":
            r = W1(offs)
        elif nm == "W1strict":
            r = W1(offs, strict=True)
        elif nm == "W3":
            r = W3(offs)
        elif nm == "W4":
            r = W4(offs)[0]
        elif nm == "W2":
            r = W4(offs, with_sockets=False)[0]
        elif nm == "W4weak":
            r = W4(offs, start="above")[0]
        elif nm == "W4m1":
            r = W4(offs, margin=1)[0]
        else:
            continue
        print(f"({nm} own) {r}  [{time.time()-t1:.1f}s RSS {rss():.0f}MB]", flush=True)
