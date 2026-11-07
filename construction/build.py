"""Build TILE instances from NAE-3SAT formulas (construction of theorem T), with a BFS wire router.

Logical frame = spin frame: background corners carry +1 (bit 0); every corner of the network of variable v carries
bit b_v; corners of a clause's private network carry a free bit.

Cell types:  background F21 (rigid);  wire F41 (domino hinge);  fan-out F42 (face hinge);  clause F22.
"""
import itertools
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import CORNERS, class_name  # noqa: E402
from tiles import (DIR_TO_K, DIRS, FULL, W_BACKGROUND, add, all_cells, coupling_with_gs, dir_between,  # noqa: E402
                   fanout_coupling, gs, is_cell, mask, wire_coupling, ALLW, GS_OF)

COLS = [(0, 0), (1, 0), (1, 1), (0, 1)]
SIDE = {(0, 0): (-2, 0, 0), (1, 0): (0, -2, 0), (1, 1): (2, 0, 0), (0, 1): (0, 2, 0)}


def vsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Instance:
    def __init__(self, L):
        self.L = L
        self.role = {}      # cell -> role string
        self.netk = {}      # cell -> {corner k: network id}
        self.W = {}         # cell -> coupling (gadget cells only; background filled at the end)
        self.reserved = set()

    def reserve(self, c, role):
        c = tuple(x % self.L for x in c)
        assert is_cell(c), c
        assert c not in self.reserved, ("cell used twice", c, role, self.role.get(c))
        self.reserved.add(c)
        self.role[c] = role
        self.netk.setdefault(c, {})
        return c

    def link(self, c1, c2, net):
        """Declare the corner shared by adjacent cells c1, c2 a corner of network `net`."""
        a = dir_between(c1, c2, self.L)
        assert a is not None, ("not adjacent", c1, c2)
        k1 = DIR_TO_K[a]
        k2 = DIR_TO_K[tuple(-x for x in a)]
        for c, k in ((c1, k1), (c2, k2)):
            old = self.netk[c].get(k)
            assert old is None or old == net, ("corner relabelled", c, k, old, net)
            self.netk[c][k] = net


# ---------------------------------------------------------------- clause cluster
def nae_clause_coupling(var_cols, private_col, negs):
    """F22 coupling, vertical columns locked equal, with  exists b_p : R  <=>  NAE(literals)."""
    kcol = {c: (c[0] + 2 * c[1], c[0] + 2 * c[1] + 4) for c in COLS}
    for j in range(ALLW.shape[0]):
        W = tuple(int(x) for x in ALLW[j])
        G = GS_OF[j]
        if len(G) != 8:
            continue
        ok = True
        rel = set()
        for m in G:
            bits = {}
            for c in COLS:
                b0, b1 = (m >> kcol[c][0]) & 1, (m >> kcol[c][1]) & 1
                if b0 != b1:
                    ok = False
                    break
                bits[c] = b0
            if not ok:
                break
            rel.add(tuple(bits[c] for c in var_cols))
        if not ok:
            continue
        want = {t for t in itertools.product([0, 1], repeat=3)
                if len({t[i] ^ negs[i] for i in range(3)}) > 1}
        if rel == want:
            assert class_name(W) == "F22"
            return W
    raise ValueError("no clause coupling")


def free_face(L, P, dN1, dN2, forbidden_cells):
    """Face of cell P containing corner directions dN1, dN2 whose other two corners avoid forbidden cells."""
    for ax in range(3):
        if dN1[ax] != dN2[ax]:
            continue
        s = dN1[ax]
        face = [a for a in DIRS if a[ax] == s]
        others = [a for a in face if a not in (dN1, dN2)]
        if all(add(P, a, L) not in forbidden_cells for a in others):
            return face, others
    raise ValueError("no free face")


def place_clause(inst, u, clause_id, lits, private_col=(0, 0)):
    """lits: list of 3 (var, neg).  Returns ports {var: (P, Q1, o1, Q2, o2)} for the three variable columns."""
    L = inst.L
    var_cols = [c for c in COLS if c != private_col]
    u = inst.reserve(u, f"clause{clause_id}")
    cells_near = {}
    N = {}
    for c in COLS:
        x, y = c
        N[c, -1] = add(u, (2 * x - 1, 2 * y - 1, -1), L)
        N[c, +1] = add(u, (2 * x - 1, 2 * y - 1, +1), L)
    P = {c: add(u, SIDE[c], L) for c in COLS}
    cluster = {u} | set(N.values()) | set(P.values())
    ports = {}
    netname = {}
    for c in COLS:
        if c == private_col:
            netname[c] = f"priv{clause_id}"
        else:
            v, neg = lits[var_cols.index(c)]
            netname[c] = f"x{v}"
    for c in COLS:
        net = netname[c]
        inst.reserve(N[c, -1], f"wireN{clause_id}")
        inst.reserve(N[c, +1], f"wireN{clause_id}")
        inst.reserve(P[c], f"side{clause_id}")
        for s in (-1, +1):
            inst.link(u, N[c, s], net)
            inst.link(N[c, s], P[c], net)
    for c in COLS:
        net = netname[c]
        dN1 = dir_between(P[c], N[c, +1], L)
        dN2 = dir_between(P[c], N[c, -1], L)
        if c == private_col:
            inst.W[P[c]] = wire_coupling(DIR_TO_K[dN1], DIR_TO_K[dN2])
            inst.role[P[c]] = f"wireP{clause_id}"
            continue
        face, others = free_face(L, P[c], dN1, dN2, cluster)
        inst.W[P[c]] = fanout_coupling([DIR_TO_K[a] for a in face])
        o1, o2 = others
        Q1 = inst.reserve(add(P[c], o1, L), f"wireQ{clause_id}")
        Q2 = inst.reserve(add(P[c], o2, L), f"wireQ{clause_id}")
        inst.link(P[c], Q1, net)
        inst.link(P[c], Q2, net)
        ports[net] = (P[c], Q1, o1, Q2, o2)
    negs = [neg for (v, neg) in lits]
    inst.W[u] = nae_clause_coupling(var_cols, private_col, negs)
    for c in COLS:
        for s in (-1, +1):
            n = N[c, s]
            ks = sorted(inst.netk[n])
            inst.W[n] = wire_coupling(*ks)
    return ports


# ---------------------------------------------------------------- router
def torus_dist(a, b, L):
    return max(min((x - y) % L, (y - x) % L) for x, y in zip(a, b))


PROTECT_R = 2
ALLOW_CONTACT = True


def route(inst, start_cell, start_in, goal_cell, goal_out, net, keepout=(), protect=(), strict=None):
    """protect: cells (open ports of other networks) whose distance-PROTECT_R neighbourhood the path must avoid."""
    L = inst.L
    banned = set()
    for q in protect:
        for d in itertools.product(range(-PROTECT_R, PROTECT_R + 1), repeat=3):
            banned.add(add(q, d, L))
    banned -= {start_cell, goal_cell}
    for _ in range(200):
        path = _route(inst, start_cell, start_in, goal_cell, goal_out, net, keepout, banned, strict)
        if path is None or len(set(path)) == len(path):
            return path
        seen = set()
        for c in path:
            if c in seen:
                banned.add(c)
            seen.add(c)
    return None


def _route(inst, start_cell, start_in, goal_cell, goal_out, net, keepout, banned, strict=None):
    """BFS for a wire path c_1..c_r with c_0 = start_cell (entered with step start_in), c_{r+1} = goal_cell,
    whose step into goal_cell, a, satisfies a . goal_out = -1 (goal_out = step leaving goal_cell).
    Consecutive steps must have dot product -1 (domino rule).  Interior cells are unused, and every used cell
    adjacent to them is a path end or carries only network `net` (contacts with other networks are avoided to keep
    routes apart; contacts are harmless for correctness, see structural_checks)."""
    L = inst.L
    res = inst.reserved

    def nets_of(r):
        return set(inst.netk[r].values())

    def ok(c):
        if c in res or c in banned:
            return False
        if torus_dist(c, start_cell, L) > 2 and torus_dist(c, goal_cell, L) > 2:
            for (u, rx, rz) in keepout:
                if max(torus_dist(c[:2] + (0,), u[:2] + (0,), L), 0) <= rx and torus_dist((0, 0, c[2]), (0, 0, u[2]), L) <= rz:
                    return False
        for a in DIRS:
            r = add(c, a, L)
            if r not in res or r == start_cell or r == goal_cell:
                continue
            if strict is not None:
                near = torus_dist(c, start_cell, L) <= 2 or torus_dist(c, goal_cell, L) <= 2
                if not (r in strict or (near and nets_of(r) == {net})):
                    return False
            elif nets_of(r) != {net} and not (ALLOW_CONTACT and inst.role[r].startswith("wire_x")):
                return False
        return True

    start = (start_cell, start_in)
    prev = {start: None}
    dq = deque([start])
    while dq:
        c, a_in = dq.popleft()
        for a in DIRS:
            if dot(a, a_in) != -1:
                continue
            n = add(c, a, L)
            if n == goal_cell and c != start_cell and dot(a, goal_out) == -1:
                path = [c]
                st = (c, a_in)
                while prev[st] is not None:
                    st = prev[st]
                    path.append(st[0])
                path = path[::-1][1:]  # drop start_cell
                return path
            if (n, a) in prev or not ok(n):
                continue
            prev[(n, a)] = (c, a_in)
            dq.append((n, a))
    return None


def build(L, clauses, centers, order=None):
    """clauses: list of [(var, neg)]*3.  centers: even cells.  Every variable's network is a main loop through the
    side cells P of its occurrences (in clause order) plus the 4-cycles at the clauses."""
    inst = Instance(L)
    keepout = [(tuple(x % L for x in u), 4, 3) for u in centers]
    occ = {}
    open_ports = {}
    for j, (cl, u) in enumerate(zip(clauses, centers)):
        ports = place_clause(inst, u, j, cl)
        for (v, neg) in cl:
            occ.setdefault(v, []).append(ports[f"x{v}"])
            P, Q1, _, Q2, _ = ports[f"x{v}"]
            open_ports[Q1] = f"x{v}"
            open_ports[Q2] = f"x{v}"
    for v, plist in sorted(occ.items()):
        net = f"x{v}"
        k = len(plist)
        for i in range(k):
            P1, _, _, Q2, o2 = plist[i]
            P2, Q1, o1, _, _ = plist[(i + 1) % k]
            protect = [q for q, nt in open_ports.items() if nt != net]
            path = route(inst, Q2, o2, Q1, tuple(-x for x in o1), net, keepout, protect)
            if path is None:
                raise RuntimeError(f"routing failed for variable {v} segment {i}")
            open_ports.pop(Q2, None)
            open_ports.pop(Q1, None)
            seq = [Q2] + path + [Q1]
            for c in path:
                inst.reserve(c, f"wire_x{v}")
            for c1, c2 in zip(seq, seq[1:]):
                inst.link(c1, c2, net)
    # wire couplings for Q cells and routed cells
    for c, role in inst.role.items():
        if c in inst.W:
            continue
        ks = sorted(inst.netk[c])
        assert len(ks) == 2, (c, role, ks)
        inst.W[c] = wire_coupling(*ks)
    Wall = {}
    for c in all_cells(L):
        Wall[c] = inst.W.get(c, W_BACKGROUND)
    return inst, Wall


# ---------------------------------------------------------------- structural checks (hypotheses of the lemmas)
def structural_checks(inst, Wall):
    L = inst.L
    out = {}
    # every corner gets one label, consistent between its two cells
    label = {}
    for c in Wall:
        for k in range(8):
            lab = inst.netk.get(c, {}).get(k, "B")
            n = add(c, DIRS[k], L)
            kk = DIR_TO_K[tuple(-x for x in DIRS[k])]
            lab2 = inst.netk.get(n, {}).get(kk, "B")
            assert lab == lab2, ("inconsistent corner label", c, k, lab, lab2)
            label[c, k] = lab
    # cell relation, read in the logical frame, has the designed type
    types = {}
    for c, W in Wall.items():
        G = gs(W)
        Bm = mask([k for k in range(8) if label[c, k] == "B"])
        cls = class_name(W)
        types[cls] = types.get(cls, 0) + 1
        nets = sorted({label[c, k] for k in range(8)} - {"B"})
        if not nets:
            assert G == frozenset({0, FULL}), ("background not rigid", c)
            continue
        if inst.role[c].startswith("clause"):
            assert Bm == 0
            continue
        # every ground state is constant on B corners
        for m in G:
            assert (m & Bm) in (0, Bm), ("B corners not locked", c)
        # with B corners = 0 the states are exactly: network corners all 0 or all 1
        Nm = FULL ^ Bm
        rest = {m for m in G if m & Bm == 0}
        assert rest == {0, Nm}, ("non-clause gadget not an equality on its network corners", c, inst.role[c])
    out["class_counts"] = types
    # B-graph connectivity: cells with a B corner, joined through shared B corners
    Bcells = [c for c in Wall if any(label[c, k] == "B" for k in range(8))]
    seen = {Bcells[0]}
    dq = deque([Bcells[0]])
    while dq:
        c = dq.popleft()
        for k in range(8):
            if label[c, k] == "B":
                n = add(c, DIRS[k], L)
                if n not in seen:
                    seen.add(n)
                    dq.append(n)
    out["B_graph_connected"] = len(seen) == len(Bcells)
    # each network connected through its corners
    nets = {}
    for (c, k), lab in label.items():
        if lab != "B":
            nets.setdefault(lab, set()).add(c)
    conn = {}
    for net, cells in nets.items():
        st = next(iter(cells))
        seen = {st}
        dq = deque([st])
        while dq:
            c = dq.popleft()
            for k in range(8):
                if label[c, k] == net:
                    n = add(c, DIRS[k], L)
                    if n not in seen:
                        seen.add(n)
                        dq.append(n)
        conn[net] = len(seen) == len(cells)
    out["networks_connected"] = all(conn.values())
    out["n_networks"] = len(nets)
    return out, label
