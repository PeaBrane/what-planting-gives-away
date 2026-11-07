"""Layout A (elementary, proof of theorem T) and expansion of coarse wirings into a fine linked TILE instance.

Coarse objects:
  * clause j gets a patch with coarse origin O_j = (6 alpha, 4 beta, 0); its cluster region is the coarse blocks
    O_j + [0,6) x [0,4) x {-KCL+1..0}; its six sockets are the coarse vertices O_j + (2a, 2b, 1);
  * each loop edge e is a directed coarse path from the socket of a + port to the socket of a - port:
    a riser at the + socket up to height h_e, a route in the plane z = h_e whose interior points all have an odd
    coordinate (so they miss every socket column), and a descender at the - socket.
Expansion: each coarse vertex of a path gets the oriented block template for (entry face, exit face); the cluster
template is translated into each patch.  Loop edges of variable v (occurrences o_1..o_k in clause order):
o_t's + port -> o_{t+1}'s - port (indices mod k).
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
from blocks import S, vadd, vsc, vsub  # noqa: E402


def patch_origins(m):
    cols = math.ceil(math.sqrt(m))
    return [(6 * (j % cols), 4 * (j // cols), 0) for j in range(m)]


def loop_edges(clauses):
    """Returns [(var, (j, i) of the + port, (j', i') of the - port)]."""
    occ = {}
    for j, cl in enumerate(clauses):
        for i, (v, _) in enumerate(cl):
            occ.setdefault(v, []).append((j, i))
    edges = []
    for v, lst in sorted(occ.items()):
        k = len(lst)
        for t in range(k):
            edges.append((v, lst[t], lst[(t + 1) % k]))
    return edges


def plane_route(p, q):
    """Points of the plane route from socket column p=(x1,y1) to q=(x2,y2) (both even); interior points have an
    odd coordinate."""
    (x1, y1), (x2, y2) = p, q
    lane = y1 + 1 if y2 >= y1 else y1 - 1
    pts = [(x1, y1), (x1 + 1, y1)]

    def go(a, b):
        while pts[-1] != (a, b):
            x, y = pts[-1]
            if x != a:
                x += 1 if a > x else -1
            else:
                y += 1 if b > y else -1
            pts.append((x, y))

    go(x1 + 1, lane)
    go(x2 + 1, lane)
    go(x2 + 1, y2)
    pts.append((x2, y2))
    assert len(set(pts)) == len(pts), (p, q, pts)
    for a, b in zip(pts, pts[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    for pt in pts[1:-1]:
        assert pt[0] % 2 or pt[1] % 2
    return pts


def layout_A(clauses):
    lk_t, assign, wires_t, u_t = B.cluster_template()
    origins = patch_origins(len(clauses))
    edges = loop_edges(clauses)
    routes = []
    for v, (j, i), (j2, i2) in edges:
        a, b = assign[i, 1]
        a2, b2 = assign[i2, -1]
        p = (origins[j][0] + 2 * a, origins[j][1] + 2 * b)
        q = (origins[j2][0] + 2 * a2, origins[j2][1] + 2 * b2)
        routes.append((p, q, plane_route(p, q)))
    # greedy heights: edges sharing a height must have vertex-disjoint plane routes
    heights = []
    level_pts = {}
    for p, q, pts in routes:
        h = 2
        while level_pts.get(h, set()) & set(pts):
            h += 1
        level_pts.setdefault(h, set()).update(pts)
        heights.append(h)
    paths = []
    for (p, q, pts), h in zip(routes, heights):
        path = [(p[0], p[1], z) for z in range(1, h)]
        path += [(x, y, h) for x, y in pts]
        path += [(q[0], q[1], z) for z in range(h - 1, 0, -1)]
        paths.append(path)
    check_coarse(paths, origins)
    return origins, edges, paths


def check_coarse(paths, origins):
    """Coarse wiring lemma hypotheses: paths vertex-disjoint, unit steps, start/end at sockets (z = 1), all other
    vertices at z >= 1 and outside cluster regions."""
    seen = set()
    for path in paths:
        assert path[0][2] == 1 and path[-1][2] == 1
        for a, b in zip(path, path[1:]):
            assert sum(abs(x - y) for x, y in zip(a, b)) == 1, (a, b)
        for g in path:
            assert g[2] >= 1
            assert g not in seen, ("coarse vertex used twice", g)
            seen.add(g)
    return True


def face_of(d):
    i = next(j for j in range(3) if d[j] != 0)
    return (i, d[i])


def expand(clauses, origins, edges, paths):
    """Fine linked instance (unbounded coordinates) and its couplings."""
    lk_t, assign, wires_t, u_t = B.cluster_template()
    T = B.templates()
    lk = B.Linked()
    negs = {}
    for j, (cl, O) in enumerate(zip(clauses, origins)):
        shift = vsc(O, S)
        names = {"priv": f"p{j}"}
        for i, (v, ng) in enumerate(cl):
            names[f"lit{i}"] = f"x{v}"
        for c in lk_t.used:
            role = lk_t.role[c]
            lk.use(vadd(c, shift), f"clause{j}" if role.startswith("clause") else f"{role}@{j}")
        for c, d in lk_t.net.items():
            for a, nt in d.items():
                c2 = vadd(c, a)
                if c < c2:
                    lk.link(vadd(c, shift), vadd(c2, shift), names[nt])
        negs[str(j)] = tuple(ng for (_, ng) in cl)
    for (v, (j, i), (j2, i2)), path in zip(edges, paths):
        cells = []
        for t, g in enumerate(path):
            f_in = (2, -1) if t == 0 else face_of(vsub(path[t - 1], g))
            f_out = (2, -1) if t == len(path) - 1 else face_of(vsub(path[t + 1], g))
            C = vsc(g, S)
            cells += [vadd(C, c) for c in T[f_in, f_out]]
        net = f"x{v}"
        lk.wire(cells, net, f"wire_x{v}")
        start = vadd(wires_t[i, 1][-1], vsc(origins[j], S))
        end = vadd(wires_t[i2, -1][-1], vsc(origins[j2], S))
        lk.link(start, cells[0], net)
        lk.link(cells[-1], end, net)
    B.couplings_for(lk, negs)
    return lk


def to_torus(lk, margin_blocks=1):
    """Translate by a multiple of S (keeps the block structure) into [0, L)^3; L a multiple of S."""
    lo = [min(c[j] for c in lk.used) for j in range(3)]
    hi = [max(c[j] for c in lk.used) for j in range(3)]
    # Blocks spanned per axis after aligning lo down to a block boundary (an earlier version used
    # S * ceil(ext / S), which ignored lo mod S and could push top cells into the margin).
    L = S * (max(math.floor(h / S) - math.floor(l / S) + 1 for l, h in zip(lo, hi)) + 2 * margin_blocks)
    shift = tuple(S * (margin_blocks - math.floor(l / S)) for l in lo)
    Wc, netc = {}, {}
    for c in lk.used:
        c2 = tuple(x + s for x, s in zip(c, shift))
        assert all(S * margin_blocks <= x < L - S * margin_blocks for x in c2), ("gadget cell in the margin", c2, L)
        assert c2 not in Wc
        Wc[c2] = lk.W[c]
        netc[c2] = {B.DIR_TO_K[a]: nt for a, nt in lk.net[c].items()}
    return L, Wc, netc, shift
