"""window-local worst-case checks for global anchoring of G46 layout A (S = 12).  Non-periodic, no torus.
Slab layer = coarse z in {-1, 0} (the only layer that can hold cluster-region blocks in layout A).
Worst case for a window: every slab-layer block of the window that is not explicitly kept is ABSENT (neither foam
nor region cells).  By monotonicity (more F42 cells only merge lock components and only enlarge anchored sets),
a connectivity or pinning fact proved in the worst case holds in every layout-A instance.
Refs of a block = corners of its foam cells lying in the periodic foam giant (g46_local2.L1), as block offsets.
(W1) z_b in -4..3, j in x,y,z: window b+[-2,2]^3, kept = non-slab blocks + {b, b+e_j}: refs(b) and refs(b+e_j)
     all in ONE window component.
(W2) cluster region R (6x4x2 blocks) + 2-block margin; slab blocks outside R absent; anchored start = components
     of refs of the present non-slab blocks; propagated anchoring must pin all cluster F6 cells.
(W3) each of the 30 templates in block b, z_b in 1..3, window b+[-2,2]^3 with slab absent, block interior
     background added; anchored start as in (W2); all template F6 cells pinned."""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import layout46 as G  # noqa: E402
import templates46 as T6  # noqa: E402
from g46_local import DUF, plate_locks  # noqa: E402
from g46_local3 import block_foam, corner, iscell, lock  # noqa: E402
from tiles import DIR_TO_K, mask  # noqa: E402
from anchor46 import giant_offsets  # noqa: E402

S, H = B.S, B.H
SLAB = (-1, 0)


def present_blocks(centre, rad, keep=()):
    out = []
    for d in itertools.product(range(-rad, rad + 1), repeat=3):
        g = tuple(c + x for c, x in zip(centre, d))
        if g[2] in SLAB and g not in keep:
            continue
        out.append(g)
    return out


CORE = os.environ.get("CORE", "1") == "1"


def refs(g, offs):
    """Core refs (default): giant refs strictly inside the block (not on a face plane shared with a neighbour)."""
    if CORE:
        offs = [d for d in offs if all(-H + 1 <= x <= H - 1 for x in d)]
    return [tuple(S * g[j] + d[j] for j in range(3)) for d in offs]


def iface(b, b2, offs):
    """Interface set I(b, b2) of adjacent blocks: periodic-giant refs in the two facing shell layers."""
    j = next(i for i in range(3) if b[i] != b2[i])
    lo, hi = (b, b2) if b[j] < b2[j] else (b2, b)
    out = [tuple(S * lo[i] + d[i] for i in range(3)) for d in offs if d[j] >= H - 1]
    out += [tuple(S * hi[i] + d[i] for i in range(3)) for d in offs if d[j] <= -H + 1]
    return out


def nbrs(g):
    for j in range(3):
        for s in (1, -1):
            yield tuple(g[i] + (s if i == j else 0) for i in range(3))


def robust(g, g2):
    """Robust interfaces: vertical ones between non-slab blocks, and lateral ones away from the slab's two
    neighbouring layers (z = -2 and z = 1), whose lower/upper parts lean on slab-layer cells."""
    if g[2] in SLAB or g2[2] in SLAB:
        return False
    return g[2] != g2[2] or g[2] not in (-2, 1)


def present_ifaces(blocks_present, offs, z_ok=lambda z: True):
    P = set(blocks_present)
    out = []
    for g in P:
        for g2 in nbrs(g):
            if g2 in P and g < g2 and z_ok(g[2]) and z_ok(g2[2]) and robust(g, g2):
                out += iface(g, g2, offs)
    return out


def W1(offs):
    """(W1*) for non-slab b: all interfaces I(b, b') with present (non-slab) neighbours b' in ONE window component;
    slab blocks of the window absent."""
    res = {}
    for zb in (-4, -3, -2, 1, 2, 3):
        b = (0, 0, zb)
        uf = DUF()
        for g in present_blocks(b, 2):
            lock(uf, block_foam(g))
        vs = [v for b2 in nbrs(b) if b2[2] not in SLAB and robust(b, b2) for v in iface(b, b2, offs)]
        res[zb] = (len({uf.find(v) for v in vs}), len(vs))
    bad = {k: v for k, v in res.items() if v[0] != 1}
    return dict(windows=len(res), comps_and_refs=res, failing=bad)


def propagate(uf, anch, f6, links):
    pending = dict(f6)
    while pending:
        newly = []
        for c, D in pending.items():
            Pm = 0
            for k in range(8):
                if k not in links[c] and uf.find(corner(c, k)) in anch:
                    Pm |= 1 << k
            if G.self_pinnable(D, Pm) is not None:
                newly.append(c)
        if not newly:
            break
        for c in newly:
            del pending[c]
            for k in range(8):
                if k not in links[c]:
                    anch.add(uf.find(corner(c, k)))
    return sorted(pending)


def W2(offs, margin=2):
    lk, assign, wires, u = G.cluster_template46()
    R = B.cluster_region((0, 0, 0))
    outside = [c for c in lk.used if not B.in_region(c, R)]
    rblocks = {(x, y, z) for x in range(6) for y in range(4) for z in SLAB}
    uf = DUF()
    nonslab = []
    for g in itertools.product(range(-margin, 6 + margin), range(-margin, 4 + margin),
                               range(-1 - margin, 1 + margin)):
        if g[2] in SLAB:
            continue
        nonslab.append(g)
        lock(uf, block_foam(g))
    region_cells = [c for c in itertools.product(*[range(lo, hi) for lo, hi in R]) if iscell(c) and c not in lk.used]
    lock(uf, region_cells)
    links, f6, fan = {}, {}, []
    for c in lk.used:
        r = lk.role[c]
        links[c] = {DIR_TO_K[a] for a in lk.net[c]}
        if r.startswith("clause"):
            continue
        if r.startswith("side"):
            fan.append((c, mask(links[c])))
        else:
            f6[c] = None
    for ij, w in wires.items():
        o = w[-1]
        tau = ij[1]
        step = B.cross_step((2, 1), tau) if tau == 1 else B.neg(B.cross_step((2, 1), tau))
        links[o].add(DIR_TO_K[step])
    for c in f6:
        assert len(links[c]) == 2, (c, links[c])
        f6[c] = mask(links[c])
    for c, m in fan:
        W = B.coupling_with_gs({0, m, 0xFF ^ m, 0xFF}, "F42")
        for j, k in plate_locks(c, W):
            uf.union(corner(c, j), corner(c, k))
    anch = {uf.find(v) for v in present_ifaces(nonslab, offs)}
    unp = propagate(uf, anch, f6, links)
    return dict(margin_blocks=margin, cluster_cells_outside_region=len(outside), region_bg_cells=len(region_cells),
                f6_cells=len(f6), fanouts=len(fan), anchored_start=len(anch),
                unpinned=len(unp), example=unp[:3])


def W3(offs):
    T = T6.templates46()
    fails, n = {}, 0
    for zb in (1, 2, 3):
        b = (0, 0, zb)
        C = tuple(S * x for x in b)
        uf0 = DUF()
        pres = present_blocks(b, 2)
        for g in pres:
            lock(uf0, block_foam(g))
        for (f1, f2), tc in sorted(T.items()):
            n += 1
            uf = DUF()
            uf.p = dict(uf0.p)
            cells_t = [B.vadd(C, c) for c in tc]
            idx = set(cells_t)
            links = {}
            for t, c in enumerate(cells_t):
                ks = set()
                ks.add(DIR_TO_K[B.vsub(cells_t[t - 1], c)] if t > 0 else
                       DIR_TO_K[B.neg(B.cross_step(f1, B.entry_tau(f1)))])
                ks.add(DIR_TO_K[B.vsub(cells_t[t + 1], c)] if t < len(cells_t) - 1 else
                       DIR_TO_K[B.cross_step(f2, f2[1])])
                links[c] = ks
            interior = [B.vadd(C, d) for d in itertools.product(range(-H + 1, H - 1), repeat=3)
                        if iscell(B.vadd(C, d)) and B.vadd(C, d) not in idx]
            lock(uf, interior)
            anch = {uf.find(v) for v in present_ifaces(pres, offs)}
            f6 = {c: mask(links[c]) for c in cells_t}
            unp = propagate(uf, anch, f6, links)
            if unp:
                fails[zb, f1, f2] = len(unp)
    return dict(checks=n, failing=fails)


def W4(offs, margin=2):
    """(W4) region R + its six socket blocks (riser template T[(2,-1),(2,1)] at + ports, descender T[(2,1),(2,-1)]
    at - ports, block interiors as background); window x,y in R +- margin, z in -1-margin..3; slab blocks outside R
    absent; anchored start = robust interfaces; region F6 and socket-template F6 cells propagated jointly."""
    from tiles import DIRS
    lk, assign, wires, u = G.cluster_template46()
    T = T6.templates46()
    R = B.cluster_region((0, 0, 0))
    uf = DUF()
    sockets = {}
    for (i, pol), (a, b) in assign.items():
        sockets[(2 * a, 2 * b, 1)] = (i, pol)
    assert len(sockets) == 6
    pres = []
    for g in itertools.product(range(-margin, 6 + margin), range(-margin, 4 + margin), range(-1 - margin, 4)):
        if g[2] in SLAB:
            continue
        pres.append(g)
        lock(uf, block_foam(g))
    used = set(lk.used)
    region_cells = [c for c in itertools.product(*[range(lo, hi) for lo, hi in R]) if iscell(c) and c not in used]
    lock(uf, region_cells)
    links, f6, fan = {}, {}, []
    for c in lk.used:
        r = lk.role[c]
        links[c] = {DIR_TO_K[a] for a in lk.net[c]}
        if r.startswith("clause"):
            continue
        if r.startswith("side"):
            fan.append((c, mask(links[c])))
        else:
            f6[c] = None
    for c, m in fan:
        W = B.coupling_with_gs({0, m, 0xFF ^ m, 0xFF}, "F42")
        for j, k in plate_locks(c, W):
            uf.union(corner(c, j), corner(c, k))
    joined = 0
    for g, (i, pol) in sockets.items():
        f1, f2 = ((2, -1), (2, 1)) if pol == 1 else ((2, 1), (2, -1))
        C = tuple(S * x for x in g)
        cells_t = [B.vadd(C, c) for c in T[f1, f2]]
        idx = set(cells_t)
        for t, c in enumerate(cells_t):
            ks = set()
            ks.add(DIR_TO_K[B.vsub(cells_t[t - 1], c)] if t > 0 else DIR_TO_K[B.neg(B.cross_step(f1, B.entry_tau(f1)))])
            ks.add(DIR_TO_K[B.vsub(cells_t[t + 1], c)] if t < len(cells_t) - 1 else DIR_TO_K[B.cross_step(f2, f2[1])])
            links[c] = ks
            f6[c] = None
        # the bottom cell of the socket template must link to the top cell of the cluster wire of this port
        bot = cells_t[0] if pol == 1 else cells_t[-1]
        w = wires[i, pol][-1]
        nb_ = [B.vadd(bot, DIRS[k]) for k in links[bot]]
        assert w in nb_, ("socket template does not meet the cluster wire", g, bot, w, nb_)
        joined += 1
        interior = [B.vadd(C, d) for d in itertools.product(range(-H + 1, H - 1), repeat=3)
                    if iscell(B.vadd(C, d)) and B.vadd(C, d) not in idx]
        lock(uf, interior)
    for ij, w in wires.items():
        o = w[-1]
        tau = ij[1]
        step = B.cross_step((2, 1), tau) if tau == 1 else B.neg(B.cross_step((2, 1), tau))
        links[o].add(DIR_TO_K[step])
    for c in f6:
        assert len(links[c]) == 2, (c, links[c])
        f6[c] = mask(links[c])
    anch = {uf.find(v) for v in present_ifaces(pres, offs)}
    unp = propagate(uf, anch, f6, links)
    return dict(sockets_joined=joined, f6_cells=len(f6), anchored_start=len(anch), unpinned=len(unp), example=unp[:3])


if __name__ == "__main__":
    t0 = time.time()
    offs = giant_offsets()
    print(f"refs per block: {len(offs)}", flush=True)
    for nm, fn in (("W1", W1), ("W2", W2), ("W3", W3), ("W4", W4)):
        if nm not in (sys.argv[1:] or ["W1", "W2", "W3", "W4"]):
            continue
        t1 = time.time()
        print(f"({nm})", fn(offs), f"[{time.time()-t1:.1f}s]", flush=True)
