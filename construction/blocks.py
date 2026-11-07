"""Explicit block layout for theorem T (replaces the heuristic fine-level router for the proof).

Coarse grid Z^3; coarse vertex g owns the block of cells u with u - S g in [-S/2, S/2)^3 (S = 8).
Canonical axis-i wire through C = S g: even cells C + 2t e_i, odd cells C + (1,1,1) + 2t e_i.
Face f = (i, sg) of a block has a boundary pair (outer, inner) of canonical cells; a block template for the
unordered face pair {f1, f2} is a domino-rule wire outer(f1), inner(f1), ..., inner(f2), outer(f2) inside the block.
The clause-cluster template occupies the 6 x 4 x K coarse blocks of its patch below coarse layer 1 and ends its
six port wires at the +z boundary pairs of blocks (2a, 2b, 0), a in {0,1,2}, b in {0,1} (the sockets).
"""
import itertools
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import CORNERS, class_name  # noqa: E402
from tiles import DIRS, DIR_TO_K, FULL, coupling_with_gs, gs, mask  # noqa: E402
from build import nae_clause_coupling  # noqa: E402

S = int(os.environ.get("TILE_S", "8"))  # block side; 8 for profile T, larger for the sparse G46 templates
assert S % 4 == 0
H = S // 2
ONE = (1, 1, 1)
FACES6 = [(i, sg) for i in range(3) for sg in (1, -1)]
KCL = 2  # coarse layers of a cluster region (coarse z = -KCL+1 .. 0)


def vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def vsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def vsc(a, k):
    return tuple(k * x for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def neg(a):
    return tuple(-x for x in a)


def unit(i):
    return tuple(1 if j == i else 0 for j in range(3))


def is_step(a):
    return all(x in (-1, 1) for x in a)


def pi(a):
    return a[0] * a[1] * a[2]


TPLUS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]  # steps of an oriented (pi = +1) wire


def a_canon(i, tau):
    """Even-to-odd step of the canonical pi=+1 straight wire travelling in direction tau e_i."""
    return next(a for a in TPLUS if a[i] == tau)


def perp(a, i):
    return tuple(0 if j == i else a[j] for j in range(3))


def boundary_pair(C, f, tau):
    """(outer, inner) cells of the canonical wire travelling tau e_i, nearest to face f = (i, sg) of block C."""
    i, sg = f
    ap = perp(a_canon(i, tau), i)
    if sg == 1:
        inner = vadd(C, vsc(unit(i), H - 2))
        outer = vadd(vadd(C, ap), vsc(unit(i), H - 1))
    else:
        outer = vadd(C, vsc(unit(i), -H))
        inner = vadd(vadd(C, ap), vsc(unit(i), 1 - H))
    return outer, inner


def cross_step(f, tau):
    """Step across face f = (i, sg) of the block at C by a wire travelling tau e_i (from the cell on the side
    the wire comes from to the cell on the other side)."""
    i, sg = f
    o, _ = boundary_pair((0, 0, 0), f, tau)
    o2, _ = boundary_pair(vsc(unit(i), S * sg), (i, -sg), tau)
    return vsub(o2, o) if tau == sg else vsub(o, o2)


def in_block(C, c):
    return all(-H <= c[j] - C[j] < H for j in range(3))


def domino_ok(a_in, a_out):
    return dot(a_in, a_out) == -1


def bfs_wire(start, a0, goal, a_goal_out, allowed, banned=frozenset()):
    """Simple wire start=c_0 (entered with step a0), ..., c_r = goal, then leaving goal with a_goal_out.
    Cells other than start must satisfy allowed(c) and not be banned. Returns the cell list or None."""
    ban = set(banned)
    for _ in range(400):
        prev = {(start, a0): None}
        dq = deque([(start, a0)])
        hit = None
        while dq and hit is None:
            c, a_in = dq.popleft()
            for a in DIRS:
                if not domino_ok(a_in, a):
                    continue
                n = vadd(c, a)
                if n == goal:
                    if domino_ok(a, a_goal_out):
                        hit = (n, a, (c, a_in))
                        break
                    continue
                if (n, a) in prev or n in ban or not allowed(n):
                    continue
                prev[(n, a)] = (c, a_in)
                dq.append((n, a))
        if hit is None:
            return None
        path = [hit[0]]
        st = hit[2]
        while st is not None:
            path.append(st[0])
            st = prev[st]
        path = path[::-1]
        if len(set(path)) == len(path):
            return path
        seen = set()
        for c in path:
            if c in seen:
                ban.add(c)
            seen.add(c)
    return None


# ---------------------------------------------------------------- block templates (oriented, pi = +1)
def entry_tau(f):
    return -f[1]


def make_template(f1, f2, C=(0, 0, 0)):
    """Wire entering the block through face f1 and leaving through face f2 (f1 != f2)."""
    t1, t2 = entry_tau(f1), f2[1]
    o1, i1 = boundary_pair(C, f1, t1)
    o2, i2 = boundary_pair(C, f2, t2)
    if f1[0] == f2[0]:  # straight: the canonical wire
        i = f1[0]
        cw = canonical_wire(C, i, t2)
        cells = sorted((c for c in block_cells(C) if c in cw), key=lambda c: c[i] * t2)
        assert cells[0] == o1 and cells[-1] == o2, (f1, f2, cells)
        return cells
    others = {o1, o2}
    path = bfs_wire(i1, vsub(i1, o1), i2, vsub(o2, i2), lambda c: in_block(C, c) and c not in others)
    if path is None:
        return None
    return [o1] + path + [o2]


def block_cells(C):
    out = []
    for d in itertools.product(range(-H, H), repeat=3):
        c = vadd(C, d)
        if c[0] % 2 == c[1] % 2 == c[2] % 2:
            out.append(c)
    return out


def canonical_wire(C, i, tau):
    ap = perp(a_canon(i, tau), i)
    out = set()
    for t in range(-S, S):
        out.add(vadd(C, vsc(unit(i), 2 * t)))
        out.add(vadd(vadd(C, ap), vsc(unit(i), 2 * t + 1)))
    return out


def check_template(f1, f2, cells, C=(0, 0, 0)):
    """Template lemma, checked: inside the block, simple, canonical boundary pairs at both ends, domino rule at
    every cell including the two outer cells with the across-face steps."""
    t1, t2 = entry_tau(f1), f2[1]
    assert all(in_block(C, c) and c[0] % 2 == c[1] % 2 == c[2] % 2 for c in cells)
    assert len(set(cells)) == len(cells)
    assert (cells[0], cells[1]) == boundary_pair(C, f1, t1)
    assert (cells[-1], cells[-2]) == boundary_pair(C, f2, t2)
    steps = [cross_step(f1, t1)] + [vsub(b, a) for a, b in zip(cells, cells[1:])] + [cross_step(f2, t2)]
    assert all(a in TPLUS for a in steps), steps
    assert all(domino_ok(a, b) for a, b in zip(steps, steps[1:]))
    return True


TEMPLATES = {}


def templates():
    if not TEMPLATES:
        for f1 in FACES6:
            for f2 in FACES6:
                if f1 == f2:
                    continue
                t = make_template(f1, f2)
                assert t is not None, ("no template", f1, f2)
                check_template(f1, f2, t)
                TEMPLATES[f1, f2] = t
    return TEMPLATES


# ---------------------------------------------------------------- linked instance
class Linked:
    def __init__(self):
        self.net = {}    # cell -> {direction: network}
        self.used = set()
        self.role = {}
        self.W = {}

    def use(self, c, role):
        assert c[0] % 2 == c[1] % 2 == c[2] % 2, ("not a cell", c)
        assert c not in self.used, ("cell used twice", c, role, self.role.get(c))
        self.used.add(c)
        self.role[c] = role
        self.net.setdefault(c, {})

    def link(self, c1, c2, nt):
        a = vsub(c2, c1)
        assert is_step(a), ("not adjacent", c1, c2)
        for c, d in ((c1, a), (c2, neg(a))):
            old = self.net.setdefault(c, {}).get(d)
            assert old is None or old == nt, ("relink", c, d, old, nt)
            self.net[c][d] = nt

    def wire(self, cells, nt, role):
        for c in cells:
            self.use(c, role)
        for c1, c2 in zip(cells, cells[1:]):
            self.link(c1, c2, nt)


COLS = [(0, 0), (1, 0), (1, 1), (0, 1)]
SIDE = {(0, 0): (-2, 0, 0), (1, 0): (0, -2, 0), (1, 1): (2, 0, 0), (0, 1): (0, 2, 0)}
VAR_COLS = [(1, 0), (1, 1), (0, 1)]  # literal positions 0, 1, 2; private column (0, 0)


def place_cluster(lk, u, tag, netname):
    """Clause cell u, 8 N wires, 4 side cells P, 6 port cells Q.  netname[c] for each column.
    Returns {(i, pol): (Q, step P->Q)} for literal position i and port polarity pol = pi(P->Q) in {+1, -1}."""
    lk.use(u, f"clause{tag}")
    N = {}
    for c in COLS:
        x, y = c
        for s_ in (-1, 1):
            N[c, s_] = vadd(u, (2 * x - 1, 2 * y - 1, s_))
    P = {c: vadd(u, SIDE[c]) for c in COLS}
    cluster = {u} | set(N.values()) | set(P.values())
    for c in COLS:
        lk.use(P[c], f"side{tag}")
        for s_ in (-1, 1):
            lk.use(N[c, s_], f"wireN{tag}")
            lk.link(u, N[c, s_], netname[c])
            lk.link(N[c, s_], P[c], netname[c])
    ports = {}
    for i, c in enumerate(VAR_COLS):
        d1, d2 = vsub(N[c, 1], P[c]), vsub(N[c, -1], P[c])
        for ax in range(3):
            if d1[ax] != d2[ax]:
                continue
            face = [a for a in DIRS if a[ax] == d1[ax]]
            oth = [a for a in face if a not in (d1, d2)]
            if all(vadd(P[c], a) not in cluster for a in oth):
                break
        else:
            raise ValueError("no free face")
        assert pi(oth[0]) == -pi(oth[1])
        for a in oth:
            q = vadd(P[c], a)
            lk.use(q, f"wireQ{tag}")
            lk.link(P[c], q, netname[c])
            ports[i, pi(a)] = (q, a)
    return ports


def cluster_region(origin):
    """Fine bounds of the cluster region whose socket (0,0) block centre is S*(origin + (0,0,1))."""
    X, Y, Z = (S * x for x in origin)
    return ((X - H, X + 6 * S - H), (Y - H, Y + 4 * S - H), (Z - H - (KCL - 1) * S, Z + H))


def in_region(c, R):
    return all(lo <= c[j] < hi for j, (lo, hi) in enumerate(R))


SOCKETS = [(a, b) for a in range(3) for b in range(2)]


def socket_pair(origin, ab, pol):
    """(outer, inner) at the +z face of coarse block origin + (2a, 2b, 0), for the oriented wire of a port of
    polarity pol: a + port's wire leaves upward (travel +z), a - port's wire arrives from above (travel -z)."""
    a, b = ab
    C0 = vadd(vsc(origin, S), (2 * a * S, 2 * b * S, 0))
    return boundary_pair(C0, (2, 1), pol)


CLUSTER_PLAN = {}


def cluster_template():
    """Find once (template coordinates, origin 0) a clause cluster plus six port wires ending at the sockets.
    Returns (lk, assign {(i, pol): socket (a, b)}, port wire cell lists from Q to the socket's outer cell, u)."""
    if CLUSTER_PLAN:
        return CLUSTER_PLAN["v"]
    origin = (0, 0, 0)
    R = cluster_region(origin)
    u = (20, 12, -4)
    names = {c: ("priv" if c == (0, 0) else f"lit{VAR_COLS.index(c)}") for c in COLS}
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            lk = Linked()
            ports = place_cluster(lk, u, "T", names)
            assign = {}
            for i in range(3):
                for pol in (1, -1):
                    assign[i, pol] = (perm[i], (pol == 1) ^ flips[i])
            pairs = {ij: socket_pair(origin, assign[ij], ij[1]) for ij in assign}
            reserved = {c for p in pairs.values() for c in p}
            wires = {}
            ok = True
            order = sorted(assign, key=lambda ij: -sum(abs(x - y) for x, y in zip(ports[ij][0], pairs[ij][1])))
            for ij in order:
                q, a_q = ports[ij]
                outer, inner = pairs[ij]
                others = reserved - {outer, inner}
                path = bfs_wire(q, a_q, inner, vsub(outer, inner),
                                lambda c: in_region(c, R) and c not in lk.used and c not in others)
                if path is None:
                    ok = False
                    break
                cells = path[1:] + [outer]
                for c in cells:
                    lk.use(c, f"port{ij}")
                for c1, c2 in zip([q] + cells, cells):
                    lk.link(c1, c2, names[VAR_COLS[ij[0]]])
                wires[ij] = [q] + cells
            if ok:
                CLUSTER_PLAN["v"] = (lk, assign, wires, u)
                return CLUSTER_PLAN["v"]
    raise RuntimeError("no cluster template")


# ---------------------------------------------------------------- couplings from link shapes (logical = spin frame)
def couplings_for(lk, clause_negs):
    """clause_negs: {clause tag: (eps_0, eps_1, eps_2)} for literal positions 0, 1, 2."""
    for c in lk.used:
        dirs = list(lk.net[c].keys())
        role = lk.role[c]
        if role.startswith("clause"):
            negs = clause_negs[role[len("clause"):]]
            lk.W[c] = nae_clause_coupling(VAR_COLS, (0, 0), list(negs))
            continue
        m = mask([DIR_TO_K[a] for a in dirs])
        if len(dirs) == 2:
            lk.W[c] = coupling_with_gs({0, m, FULL ^ m, FULL}, "F41")
        elif len(dirs) == 4:
            lk.W[c] = coupling_with_gs({0, m, FULL ^ m, FULL}, "F42")
        else:
            raise ValueError(("bad link count", c, role, dirs))


W_BG = coupling_with_gs({0, FULL}, "F21")


# ---------------------------------------------------------------- well-typedness (hypotheses of Lemma 3.1)
FACE_MASKS = {mask([k for k in range(8) if CORNERS[k][ax] == v]) for ax in range(3) for v in (0, 1)}
DOMINO_MASKS = {mask([a, b]) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1}
COLUMN_KS = [(0, 4), (1, 5), (3, 7), (2, 6)]


def check_well_typed(cells_W, labels_of, L):
    """cells_W: iterable of (cell, W); labels_of(cell) -> {corner k: network}.  Raises on any violation.
    Returns the class counts and the clause relations (as column-label tuples)."""
    counts = {}
    clauses = []
    for c, W in cells_W:
        G = gs(W)
        lab = labels_of(c)
        cls = class_name(W)
        counts[cls] = counts.get(cls, 0) + 1
        nets = set(lab.values())
        if not lab:
            assert G == frozenset({0, FULL}), ("background not rigid", c)
            continue
        if len(lab) == 8:
            cols = []
            for k0, k1 in COLUMN_KS:
                assert lab[k0] == lab[k1], ("clause column split", c)
                cols.append(lab[k0])
            assert len(set(cols)) == 4, ("clause columns not distinct", c)
            for m in G:
                for k0, k1 in COLUMN_KS:
                    assert ((m >> k0) & 1) == ((m >> k1) & 1), ("clause column not locked", c)
            rel = sorted({tuple((m >> k0) & 1 for k0, _ in COLUMN_KS) for m in G})
            clauses.append((c, cols, rel))
            continue
        assert len(nets) == 1, ("two networks in a non-clause cell", c, lab)
        m = mask(lab.keys())
        assert m in DOMINO_MASKS or m in FACE_MASKS, ("network corners neither domino nor face", c, sorted(lab))
        assert G == frozenset({0, m, FULL ^ m, FULL}), ("wrong ground-state set", c)
    return counts, clauses
