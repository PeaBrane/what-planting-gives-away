"""T+ attempt for Hamze's gallus_46 classes (F6 u F42 only), on the explicit polarity-aware block layout.

Profile G46 (logical frame = spin frame):
  background  F42, plates perpendicular to axis (u_x + u_y + u_z) mod 3 (first run: plates glue into one component);
  wire        F6, with the cell's gauge chosen so that, with its corners in the anchor component A0 at 0, its
              ground states are exactly {0, 1_D} (D its domino of link corners) -- "self-pinning";
  fan-out     F42 face hinge (as in T);
  clause      F6 whose corners split into the mixed domino matching {01, 26, 37, 45} (f6_clause_search.py), the
              private domino 45; exists d: R = NAE(literals).
Generalised network lemma (construction notes Sec. 8): if A0 is a component of the lock graph of the affine cells, every F6
wire cell C satisfies Gamma_C cap {sigma = 0 on C cap A0} = {0, 1_D}, fan-outs/clauses are as stated and every network
is connected through wire and fan-out cells, then planted <=> the clause relations are jointly satisfiable.
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
import lean_tile as LT  # noqa: E402
from blocks import S, dot, pi, vadd, vsc, vsub  # noqa: E402
from cube import CORNERS, class_name  # noqa: E402
from layout import face_of, patch_origins, plane_route  # noqa: E402
from tiles import ALLW, DIR_TO_K, DIRS, FULL, GS_OF, coupling_with_gs, mask  # noqa: E402

DOMS = [(0, 1), (2, 6), (3, 7), (4, 5)]
PRIV = 3
VAR_IDX = [0, 1, 2]


def face_mask(ax, val):
    return mask([k for k in range(8) if CORNERS[k][ax] == val])


F6_LIST = [(tuple(int(x) for x in ALLW[j]), GS_OF[j]) for j in range(ALLW.shape[0]) if len(GS_OF[j]) == 16]
assert len(F6_LIST) == 128


_BG = {}


def bg_coupling(u):
    ax = (u[0] + u[1] + u[2]) % 3
    if ax not in _BG:
        _BG[ax] = coupling_with_gs({0, face_mask(ax, 0), face_mask(ax, 1), FULL}, "F42")
    return _BG[ax]


_CL = {}


def clause_coupling46(negs):
    negs = tuple(negs)
    if negs not in _CL:
        Ds = [mask(d) for d in DOMS]
        want = {t for t in itertools.product([0, 1], repeat=3) if len({t[i] ^ negs[i] for i in range(3)}) > 1}
        for W, G in F6_LIST:
            rel = set()
            for s in range(16):
                m = 0
                for i in range(4):
                    if (s >> i) & 1:
                        m ^= Ds[i]
                if m in G:
                    rel.add(tuple((s >> i) & 1 for i in VAR_IDX))
            if rel == want:
                _CL[negs] = W
                break
        else:
            raise ValueError("no F6 clause coupling")
    return _CL[negs]


def place_cluster46(lk, u, tag, netname):
    """netname[i] for domino i.  Returns ports {(i, pol): (Q, step)} for variable dominoes i in VAR_IDX."""
    N = {}
    for i, d in enumerate(DOMS):
        for k in d:
            N[i, k] = vadd(u, DIRS[k])
    opts = []
    for d in DOMS:
        a1, a2 = DIRS[d[0]], DIRS[d[1]]
        opts.append([vadd(u, vsc(B.unit(j), 2 * a1[j])) for j in range(3) if a1[j] == a2[j]])
    for choice in itertools.product(*opts):
        if len(set(choice)) < 4:
            continue
        cluster = {u} | set(N.values()) | set(choice)
        faces = {}
        ok = True
        for i in VAR_IDX:
            P = choice[i]
            d1, d2 = vsub(N[i, DOMS[i][0]], P), vsub(N[i, DOMS[i][1]], P)
            for ax in range(3):
                if d1[ax] != d2[ax]:
                    continue
                face = [a for a in DIRS if a[ax] == d1[ax]]
                oth = [a for a in face if a not in (d1, d2)]
                if all(vadd(P, a) not in cluster for a in oth):
                    faces[i] = oth
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        outward = [vadd(choice[i], a) for i in faces for a in faces[i]]
        if len(set(outward)) < len(outward) or set(outward) & cluster:
            continue
        break
    else:
        raise ValueError("no G46 cluster plan")
    lk.use(u, f"clause{tag}")
    for i, d in enumerate(DOMS):
        lk.use(choice[i], f"side{tag}" if i != PRIV else f"wireP{tag}")
        for k in d:
            lk.use(N[i, k], f"wireN{tag}")
            lk.link(u, N[i, k], netname[i])
            lk.link(N[i, k], choice[i], netname[i])
    ports = {}
    for i in VAR_IDX:
        oth = faces[i]
        assert pi(oth[0]) == -pi(oth[1])
        for a in oth:
            q = vadd(choice[i], a)
            lk.use(q, f"wireQ{tag}")
            lk.link(choice[i], q, netname[i])
            ports[i, pi(a)] = (q, a)
    if TWO_LEVEL:
        ports = second_level(lk, ports, choice, tag, netname)
    return ports


TWO_LEVEL = True


def second_level(lk, ports, P, tag, netname):
    """Replace each fan-out's port pair (Q+, Q-) by a second F42 fan-out E adjacent to both, whose free face gives
    the new port pair (R+, R-) further out.  E and R must touch no F6 cell other than their links."""
    out = {}
    for i in VAR_IDX:
        (q1, o1), (q2, o2) = ports[i, 1], ports[i, -1]
        k = next(j for j in range(3) if o1[j] != o2[j])
        # common neighbours of Q1, Q2 satisfying the domino rule at both: P + 2 o1_j e_j, j != k
        cands = [vadd(P[i], vsc(B.unit(j), 2 * o1[j])) for j in range(3) if j != k]
        done = False
        for E in cands:
            if E in lk.used:
                continue
            d1, d2 = vsub(q1, E), vsub(q2, E)
            if not all(abs(x) == 1 for x in d1 + d2):
                continue
            for ax in range(3):
                if d1[ax] != d2[ax]:
                    continue
                face = [a for a in DIRS if a[ax] == d1[ax]]
                oth = [a for a in face if a not in (d1, d2)]
                rs = [vadd(E, a) for a in oth]
                if any(r in lk.used for r in rs):
                    continue
                def touches_f6(c, allow):
                    return [vadd(c, a) for a in DIRS if vadd(c, a) in lk.used and vadd(c, a) not in allow
                            and is_f6_role(lk.role[vadd(c, a)])]
                if touches_f6(E, {q1, q2}):
                    continue
                if any(touches_f6(r, set()) for r in rs):
                    continue
                lk.use(E, f"side{tag}")
                lk.link(q1, E, netname[i])
                lk.link(q2, E, netname[i])
                for a, r in zip(oth, rs):
                    lk.use(r, f"wireQ{tag}")
                    lk.link(E, r, netname[i])
                    out[i, pi(a)] = (r, a)
                done = True
                break
            if done:
                break
        if not done:
            raise ValueError("no second-level fan-out")
    return out


PLAN46 = {}


def f6_contacts(lk, c, f6):
    """Unlinked neighbours of cell c that are F6 cells (f6: predicate on cells)."""
    out = []
    for a in DIRS:
        r = vadd(c, a)
        if r in lk.used and a not in lk.net.get(c, {}) and f6(r):
            out.append(r)
    return out


def is_f6_role(role):
    return not role.startswith("side")  # fan-outs ("side") are F42; clause, N, P_priv, Q and port wires are F6


_PIN = {}


def self_pinnable(D, Pm):
    """Exists an F6 coupling with 0, 1_D in Gamma and Gamma cap {sigma: sigma & Pm = 0} = {0, 1_D}."""
    key = (D, Pm)
    if key not in _PIN:
        _PIN[key] = next((W for W, G in F6_LIST if 0 in G and D in G and {m for m in G if m & Pm == 0} == {0, D}),
                         None)
    return _PIN[key]


def unpinnable_cells(lk, cells):
    """F6 wire cells among `cells` that are not self-pinnable when every corner shared with a non-F6 cell or with
    background counts as anchored and every corner shared with an unlinked F6 cell is floating."""
    bad = []
    for c in cells:
        if not is_f6_role(lk.role[c]) or lk.role[c].startswith("clause") or len(lk.net[c]) < 2:
            continue
        assert len(lk.net[c]) == 2
        D = mask(DIR_TO_K[a] for a in lk.net[c])
        fl = 0
        for a in DIRS:
            r = vadd(c, a)
            if a not in lk.net[c] and r in lk.used and is_f6_role(lk.role[r]):
                fl |= 1 << DIR_TO_K[a]
        if self_pinnable(D, 0xFF ^ D ^ fl) is None:
            bad.append(c)
    return bad


def cluster_template46(max_assign=6, retries=40):
    """Contact-aware cluster template: every F6 wire cell of the cluster must be self-pinnable assuming its
    corners shared with F42 or background cells are anchored (checked after each port wire; offending cells of the
    new wire banned and the route retried)."""
    if PLAN46:
        return PLAN46["v"]
    origin = (0, 0, 0)
    R = B.cluster_region(origin)
    u = (5 * S // 2, 3 * S // 2, -S // 2)

    def extreme_R(c):
        return any(c[j] in (lo, hi - 1) for j, (lo, hi) in enumerate(R))

    names = {i: ("priv" if i == PRIV else f"lit{i}") for i in range(4)}
    tried = 0
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            tried += 1
            if tried > max_assign:
                raise RuntimeError("no G46 cluster template within the assignment budget")
            lk = B.Linked()
            ports = place_cluster46(lk, u, "T", names)
            bare_bad = unpinnable_cells(lk, list(lk.used))
            if bare_bad:
                raise RuntimeError(f"bare G46 cluster has unpinnable cells {bare_bad}")
            assign = {(i, pol): (perm[i], (pol == 1) ^ flips[i]) for i in VAR_IDX for pol in (1, -1)}
            pairs = {ij: B.socket_pair(origin, assign[ij], ij[1]) for ij in assign}
            reserved = {c for p in pairs.values() for c in p}
            wires = {}
            ok = True
            order = sorted(assign, key=lambda ij: -sum(abs(x - y) for x, y in zip(ports[ij][0], pairs[ij][1])))
            for ij in order:
                q, a_q = ports[ij]
                outer, inner = pairs[ij]
                others = reserved - {outer, inner}
                f6used = {c for c in lk.used if is_f6_role(lk.role[c])}
                near = {}
                for c in f6used:
                    for a in DIRS:
                        n = vadd(c, a)
                        if n not in lk.used:
                            near.setdefault(n, []).append(c)
                banned = set()
                path = None
                for _ in range(retries):
                    def allowed(c, q=q):
                        if c in lk.used or c in others or c in banned or not B.in_region(c, R) or extreme_R(c):
                            return False
                        return len([r for r in near.get(c, ()) if r != q]) <= 1
                    path = B.bfs_wire(q, a_q, inner, vsub(outer, inner), allowed)
                    if path is None:
                        break
                    cells = path[1:] + [outer]
                    for c in cells:
                        lk.use(c, f"port{ij}")
                    for c1, c2 in zip([q] + cells, cells):
                        lk.link(c1, c2, names[ij[0]])
                    touched = set(cells) | {r for c in cells for r in near.get(c, ())} | {q}
                    bad = unpinnable_cells(lk, touched)
                    if not bad:
                        break
                    # undo
                    for c in cells:
                        lk.used.discard(c)
                        lk.role.pop(c, None)
                        lk.net.pop(c, None)
                    lk.net[q].pop(vsub(cells[0], q), None)
                    nb = {c for c in bad if c in cells and c not in (inner, outer)}
                    nb |= {c for c in cells if any(r in bad for r in near.get(c, ())) and c not in (inner, outer)}
                    banned |= nb or {cells[len(cells) // 2]}
                    path = None
                if path is None:
                    ok = False
                    break
                wires[ij] = [q] + cells
            if ok:
                PLAN46["v"] = (lk, assign, wires, u)
                return PLAN46["v"]
    raise RuntimeError("no G46 cluster template")


def layout_A46(clauses):
    lk_t, assign, wires_t, u_t = cluster_template46()
    origins = patch_origins(len(clauses))
    occ = {}
    for j, cl in enumerate(clauses):
        for i, (v, _) in enumerate(cl):
            occ.setdefault(v, []).append((j, i))
    edges = []
    for v, lst in sorted(occ.items()):
        k = len(lst)
        for t in range(k):
            edges.append((v, lst[t], lst[(t + 1) % k]))
    routes = []
    for v, (j, i), (j2, i2) in edges:
        a, b = assign[i, 1]
        a2, b2 = assign[i2, -1]
        p = (origins[j][0] + 2 * a, origins[j][1] + 2 * b)
        q = (origins[j2][0] + 2 * a2, origins[j2][1] + 2 * b2)
        routes.append((p, q, plane_route(p, q)))
    heights, level_pts = [], {}
    for p, q, pts in routes:
        h = 2
        # G46: keep same-height routes one unit apart as well (no block-face contacts between different wires)
        while any((x + dx, y + dy) in level_pts.get(h, set()) for x, y in pts for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            h += 1
        level_pts.setdefault(h, set()).update(pts)
        heights.append(h)
    paths = []
    for (p, q, pts), h in zip(routes, heights):
        path = [(p[0], p[1], z) for z in range(1, h)]
        path += [(x, y, h) for x, y in pts]
        path += [(q[0], q[1], z) for z in range(h - 1, 0, -1)]
        paths.append(path)
    from layout import check_coarse
    check_coarse(paths, origins)
    return origins, edges, paths


def expand46(clauses, origins, edges, paths):
    lk_t, assign, wires_t, u_t = cluster_template46()
    import templates46
    T = templates46.templates46()
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
        negs[j] = tuple(ng for (_, ng) in cl)
    for (v, (j, i), (j2, i2)), path in zip(edges, paths):
        cells = []
        for t, g in enumerate(path):
            f_in = (2, -1) if t == 0 else face_of(vsub(path[t - 1], g))
            f_out = (2, -1) if t == len(path) - 1 else face_of(vsub(path[t + 1], g))
            cells += [vadd(vsc(g, S), c) for c in T[f_in, f_out]]
        net = f"x{v}"
        lk.wire(cells, net, f"wire_x{v}")
        lk.link(vadd(wires_t[i, 1][-1], vsc(origins[j], S)), cells[0], net)
        lk.link(cells[-1], vadd(wires_t[i2, -1][-1], vsc(origins[j2], S)), net)
    return lk, negs


def to_torus46(lk, negs, margin_blocks=1):
    lo = [min(c[j] for c in lk.used) for j in range(3)]
    hi = [max(c[j] for c in lk.used) for j in range(3)]
    # Blocks spanned per axis after aligning lo down to a block boundary (same fix as layout.to_torus).
    nb = max(math.floor(h / S) - math.floor(l / S) + 1 for l, h in zip(lo, hi)) + 2 * margin_blocks
    while (nb * S) % 3:  # keep the background axis pattern periodic on the torus
        nb += 1
    L = S * nb
    shift = tuple(S * (margin_blocks - math.floor(l / S)) for l in lo)
    netc, role = {}, {}
    for c in lk.used:
        c2 = tuple(x + s for x, s in zip(c, shift))
        assert all(S * margin_blocks <= x < L - S * margin_blocks for x in c2)
        netc[c2] = {DIR_TO_K[a]: nt for a, nt in lk.net[c].items()}
        role[c2] = lk.role[c]
    return L, netc, role


def assign_couplings46(L, netc, role, negs):
    """Affine cells first (background F42, fan-outs F42), anchor component A0 = component of the lock graph
    containing the most B incidences; then each F6 wire gets a self-pinning gauge.  Returns (Wc, anchor_info)."""
    Wc = {}
    for c, lab in netc.items():
        r = role[c]
        if r.startswith("clause"):
            Wc[c] = clause_coupling46(negs[int(r[len("clause"):])])
        elif len(lab) == 4:
            m = mask(lab.keys())
            Wc[c] = coupling_with_gs({0, m, FULL ^ m, FULL}, "F42")
    W_aff = lambda u: Wc.get(u) if u in Wc else (bg_coupling(u) if u not in netc else None)  # noqa: E731
    uf = LT.UF(L ** 3)
    for u in LT.all_cells(L):
        W = W_aff(u)
        if W is None or class_name(W) != "F42":
            continue
        G, affine, lks = LT.gs_info(W)
        assert affine
        for k0, k, x in lks:
            assert uf.union(LT.corner_index(L, u, k0), LT.corner_index(L, u, k), x)
    counts = {}
    for u in LT.all_cells(L):
        lab = netc.get(u, {})
        for k in range(8):
            if k not in lab:
                r, p = uf.find(LT.corner_index(L, u, k))
                counts[r] = counts.get(r, 0) + 1
    root = max(counts, key=counts.get)
    n_comp = len(counts)
    # propagated anchoring: A = lock components known to be 0 in every planted state (WLOG); a wire cell whose
    # anchored corners force {0, 1_D} pins all its other corners, whose components then join A
    anchored = {root}
    pending = [c for c in netc if c not in Wc]
    order = []
    while pending:
        progress = []
        for c in pending:
            lab = netc[c]
            assert len(lab) == 2
            D = mask(lab.keys())
            Pm = 0
            for k in range(8):
                if k in lab:
                    continue
                r, p = uf.find(LT.corner_index(L, c, k))
                if r in anchored:
                    assert p == 0
                    Pm |= 1 << k
            W = self_pinnable(D, Pm)
            if W is not None:
                Wc[c] = W
                progress.append(c)
                order.append((c, Pm))
                for k in range(8):
                    if k not in lab:
                        anchored.add(uf.find(LT.corner_index(L, c, k))[0])
        if not progress:
            c = pending[0]
            raise RuntimeError(f"{len(pending)} F6 wire cells not pinnable by propagated anchoring, e.g. {c} ({role[c]})")
        done = set(progress)
        pending = [c for c in pending if c not in done]
    stray = 0
    for u in LT.all_cells(L):
        if u in netc:
            continue
        for k in range(8):
            r, _ = uf.find(LT.corner_index(L, u, k))
            if r not in anchored:
                stray += 1
    rounds_info = dict(lock_components_with_B=n_comp, main_component_B_incidences=counts[root],
                       anchored_components=len(anchored), bg_incidences_never_anchored=stray)
    return Wc, rounds_info, uf, (anchored, order)


def check_lemma46(L, netc, role, Wc, uf, anch):
    """Hypotheses of the generalised network lemma, cell by cell from raw couplings.  Anchoring is re-derived
    independently: start from the main lock component and repeatedly add the B corners of any F6 wire cell whose
    raw ground-state set, restricted to its currently anchored corners at 0, is exactly {0, 1_D}."""
    anchored0, _ = anch
    counts = {}
    wires = []
    for u in LT.all_cells(L):
        W = Wc.get(u) or bg_coupling(u)
        cls = class_name(W)
        counts[cls] = counts.get(cls, 0) + 1
        G, _, _ = LT.gs_info(W)
        lab = netc.get(u, {})
        if not lab:
            assert cls == "F42" and 0 in G
            continue
        if role[u].startswith("clause"):
            assert cls == "F6" and len(lab) == 8
            labs = []
            for d in DOMS:
                assert lab[d[0]] == lab[d[1]]
                labs.append(lab[d[0]])
            assert len(set(labs)) == 4
            continue
        assert len(set(lab.values())) == 1
        m = mask(lab.keys())
        if len(lab) == 4:
            assert cls == "F42" and set(G) == {0, m, FULL ^ m, FULL}
            continue
        assert len(lab) == 2 and cls == "F6" and 0 in G and m in G
        wires.append(u)
    # independent anchoring fixpoint
    cnt = {}
    for u in LT.all_cells(L):
        lab = netc.get(u, {})
        for k in range(8):
            if k not in lab:
                r, _ = uf.find(LT.corner_index(L, u, k))
                cnt[r] = cnt.get(r, 0) + 1
    anchored = {max(cnt, key=cnt.get)}
    pending = set(wires)
    rounds = 0
    while pending:
        rounds += 1
        newly = []
        for u in pending:
            lab = netc[u]
            m = mask(lab.keys())
            G, _, _ = LT.gs_info(Wc[u])
            Pm = 0
            for k in range(8):
                if k not in lab and uf.find(LT.corner_index(L, u, k))[0] in anchored:
                    Pm |= 1 << k
            if {s_ for s_ in G if s_ & Pm == 0} == {0, m}:
                newly.append(u)
        assert newly, f"{len(pending)} wire cells never pinned"
        for u in newly:
            pending.discard(u)
            for k in range(8):
                if k not in netc[u]:
                    anchored.add(uf.find(LT.corner_index(L, u, k))[0])
    counts["anchoring_rounds"] = rounds
    return counts
