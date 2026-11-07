"""End-to-end check: the construction code generates the instance; independent code checks typing and decides TILE.

Usage: e2e_check.py <formulas.json> <name> [margin]
The construction code is used only for:  layout.layout_A, layout.expand, layout.to_torus, blocks.W_BG.
"""
import os
import itertools
import json
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_decider as D  # noqa: E402

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "construction"))
import blocks  # noqa: E402
import layout  # noqa: E402

FACES = [(ax, v) for ax in range(3) for v in (0, 1)]
FACE_EDGES = [[i for i, (a, b) in enumerate(D.EDGES) if D.CORN[a][ax] == v and D.CORN[b][ax] == v] for ax, v in FACES]
DOMINO = {(1 << a) | (1 << b) for a, b in D.EDGES}
FACEM = {sum(1 << k for k in range(8) if D.CORN[k][ax] == v) for ax, v in FACES}
COLUMNS = [(0, 4), (1, 5), (3, 7), (2, 6)]


def hamze_class(W):
    T = [f for f in range(6) if (-1) ** sum(W[i] < 0 for i in FACE_EDGES[f]) < 0]
    n = len(T)
    if n in (0, 6):
        return f"F{n}"
    if n == 2:
        return "F22" if FACES[T[0]][0] == FACES[T[1]][0] else "F21"
    un = [f for f in range(6) if f not in T]
    return "F42" if FACES[un[0]][0] == FACES[un[1]][0] else "F41"


def nae_sat(clauses, n):
    sols = []
    for a in itertools.product((0, 1), repeat=n):
        if all(len({a[v] ^ ng for v, ng in cl}) > 1 for cl in clauses):
            sols.append(a)
    return sols


def fixed_torus(lk, margin):
    """Block-aligned embedding: L = S * (#blocks spanned + 2 margin); same translation-by-multiples-of-S rule."""
    S, H = blocks.S, blocks.H
    lo = [min((c[j] + H) // S for c in lk.used) for j in range(3)]
    hi = [max((c[j] + H) // S for c in lk.used) for j in range(3)]
    nb = max(h - l + 1 for l, h in zip(lo, hi))
    L = S * (nb + 2 * margin)
    shift = tuple(S * (margin - l) for l in lo)
    Wc, netc = {}, {}
    for c in lk.used:
        c2 = tuple(x + s for x, s in zip(c, shift))
        assert all(0 <= x < L for x in c2) and c2 not in Wc
        Wc[c2] = lk.W[c]
        netc[c2] = {blocks.DIR_TO_K[a]: nt for a, nt in lk.net[c].items()}
    return L, Wc, netc, shift


def gamma(W):
    e = D.cell_energy_table(W)
    return frozenset(int(m) for m in (e == e.min()).nonzero()[0])


def main():
    formulas = json.load(open(sys.argv[1]))
    name = sys.argv[2]
    margin = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    cl = [[tuple(l) for l in c] for c in formulas[name]]
    run(cl, name, margin)


def run(cl, name, margin, coarse=None):
    """coarse: optional (origins, edges, paths) replacing layout_A (extra test of Lemma 4.6 on other wirings)."""
    out = {"name": name, "clauses": cl, "margin": margin, "coarse_source": "layout_A" if coarse is None else "custom"}
    t0 = time.time()
    n = 1 + max(v for c in cl for v, _ in c)
    sols = nae_sat(cl, n)
    out["n"], out["m"], out["nae_sat"], out["n_nae_solutions"] = n, len(cl), bool(sols), len(sols)

    # ---------------- generate (construction code)
    origins, edges, paths = layout.layout_A(cl) if coarse is None else coarse
    if coarse is not None:
        layout.check_coarse(paths, origins)
    lk = layout.expand(cl, origins, edges, paths)
    try:
        L, Wc, netc, shift = layout.to_torus(lk, margin)
    except AssertionError as ex:
        # an earlier to_torus sized L from the cell extent, not the block range.  Diagnose and
        # fall back to an independent block-aligned embedding (same shift rule, L from block indices).
        lo = [min(c[j] for c in lk.used) for j in range(3)]
        hi = [max(c[j] for c in lk.used) for j in range(3)]
        out["to_torus_assertion"] = str(ex)[:120]
        out["to_torus_extent_lo_hi_mod8"] = [(l, h, l % 8, h % 8) for l, h in zip(lo, hi)]
        L, Wc, netc, shift = fixed_torus(lk, margin)
        out["used_block_aligned_embedding"] = True
    out["L"], out["gadget_cells"] = L, len(Wc)
    if L > 112:
        out["skipped"] = "L > 112: decider would approach the 150 MB cap"
        print(json.dumps(out, default=str))
        return out

    # ---------------- independent coarse-wiring checks (Lemma 4.7 hypotheses, own code)
    seen, ok = set(), True
    for (v, (j, i), (j2, i2)), p in zip(edges, paths):
        O, O2 = origins[j], origins[j2]
        ok &= p[0] == (O[0] + 2 * i, O[1] + 2, 1) and p[-1] == (O2[0] + 2 * i2, O2[1], 1)
        ok &= all(sum(abs(a - b) for a, b in zip(g, h)) == 1 for g, h in zip(p, p[1:]))
        ok &= all(g[2] >= 1 for g in p) and len(set(p)) == len(p)
        ok &= not (set(p) & seen)
        seen |= set(p)
        ok &= cl[j][i][0] == v and cl[j2][i2][0] == v
    plus = sorted((j, i) for _, (j, i), _ in edges)
    minus = sorted((j2, i2) for _, _, (j2, i2) in edges)
    allports = sorted((j, i) for j in range(len(cl)) for i in range(3))
    ok &= plus == allports and minus == allports
    out["coarse_wiring_ok"] = bool(ok)
    out["max_height"] = max(g[2] for p in paths for g in p)

    # ---------------- independent typing checks (Sec. 3.1), labels from the builder's links
    Wbg = tuple(blocks.W_BG)
    G_bg = gamma(Wbg)
    out["background_class"], out["background_gamma"] = hamze_class(Wbg), sorted(G_bg)

    def cell_of(u):
        return tuple(x % L for x in u)

    vlabel = {}
    label_ok = True
    for c, lab in netc.items():
        for k, x in lab.items():
            d = D.CORN[k]
            v = cell_of((c[0] + d[0], c[1] + d[1], c[2] + d[2]))
            other = cell_of((c[0] + 2 * d[0] - 1, c[1] + 2 * d[1] - 1, c[2] + 2 * d[2] - 1))
            if netc.get(other, {}).get(7 - k) != x:
                label_ok = False          # the other cell through the vertex must carry the same label
            if vlabel.get(v, x) != x:
                label_ok = False
            vlabel[v] = x
    out["labels_consistent_vertex_property"] = label_ok
    classes = {}
    gam_cache = {}
    typing_errors = []
    clause_cells = []
    nets_adj = {}

    def uf_find(par, a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def vid(c, k):
        d = D.CORN[k]
        return cell_of((c[0] + d[0], c[1] + d[1], c[2] + d[2]))

    for c, W in Wc.items():
        W = tuple(W)
        if W not in gam_cache:
            gam_cache[W] = (gamma(W), hamze_class(W))
        G, cls = gam_cache[W]
        classes[cls] = classes.get(cls, 0) + 1
        lab = netc.get(c, {})
        labs = set(lab.values())
        if not lab:
            if G != {0, 255}:
                typing_errors.append(("unlabelled gadget cell not rigid", c))
            continue
        if len(lab) == 8:
            cols = []
            for k0, k1 in COLUMNS:
                if lab[k0] != lab[k1]:
                    typing_errors.append(("clause column split", c))
                cols.append(lab[k0])
                nets_adj.setdefault(lab[k0], []).append((vid(c, k0), vid(c, k1)))
            if len(set(cols)) != 4:
                typing_errors.append(("clause labels not distinct", c))
            if any(((m >> k0) & 1) != ((m >> k1) & 1) for m in G for k0, k1 in COLUMNS):
                typing_errors.append(("clause not column-locked", c))
            clause_cells.append((c, cols, G, cls))
            continue
        if len(labs) != 1:
            typing_errors.append(("two labels in a non-clause cell", c))
            continue
        msk = sum(1 << k for k in lab)
        shape = "domino" if msk in DOMINO else ("face" if msk in FACEM else None)
        if shape is None:
            typing_errors.append(("labelled corners neither domino nor face", c, sorted(lab)))
            continue
        if G != {0, msk, 255 ^ msk, 255}:
            typing_errors.append(("wrong Gamma for", shape, c))
        if (shape, cls) not in (("domino", "F41"), ("face", "F42")):
            typing_errors.append(("unexpected class", shape, cls, c))
        ks = sorted(lab)
        x = lab[ks[0]]
        for k in ks[1:]:
            nets_adj.setdefault(x, []).append((vid(c, ks[0]), vid(c, k)))
    classes["F21(background)"] = (L ** 3 // 4) - len(Wc) if hamze_class(Wbg) == "F21" else None
    out["class_counts_gadget_cells"] = classes
    out["typing_errors"] = typing_errors[:10]
    out["n_typing_errors"] = len(typing_errors)
    # every vertex of a gadget cell whose other cell is background must be unlabelled: covered by label check.
    # network connectivity (G_x of Sec. 3.1)
    verts_by_net = {}
    for v, x in vlabel.items():
        verts_by_net.setdefault(x, set()).add(v)
    conn = {}
    for x, vs in verts_by_net.items():
        par = {}
        for a, b in nets_adj.get(x, []):
            ra, rb = uf_find(par, a), uf_find(par, b)
            if ra != rb:
                par[ra] = rb
        roots = {uf_find(par, v) for v in vs}
        conn[x] = len(roots)
    out["networks"] = len(conn)
    out["all_networks_connected"] = all(r == 1 for r in conn.values())
    out["network_vertex_counts"] = {x: len(v) for x, v in sorted(verts_by_net.items())}
    # clause relations vs NAE of the literals
    rel_ok = len(clause_cells) == len(cl)
    for c, cols, G, cls in clause_cells:
        priv = [x for x in cols if x.startswith("p")]
        if len(priv) != 1:
            rel_ok = False
            continue
        j = int(priv[0][1:])
        lits = cl[j]
        varlabels = {f"x{v}": ng for v, ng in lits}
        if set(cols) - set(priv) != set(varlabels) or cls != "F22":
            rel_ok = False
            continue
        proj = set()
        for m in G:
            vals = {cols[i]: (m >> COLUMNS[i][0]) & 1 for i in range(4)}
            proj.add(tuple(vals[f"x{v}"] for v, _ in lits))
        want = {a for a in itertools.product((0, 1), repeat=3) if len({a[i] ^ lits[i][1] for i in range(3)}) > 1}
        rel_ok &= proj == want
        # the private network occurs in no other clause cell
        rel_ok &= sum(priv[0] in cc[1] for cc in clause_cells) == 1
    out["clause_relations_are_NAE_of_literals"] = bool(rel_ok)
    # contact census on the torus: unlinked vertex-sharing pairs of gadget cells
    cen = {}
    for c, lab in netc.items():
        for k in range(8):
            if k in lab:
                continue
            d = D.CORN[k]
            raw = (c[0] + 2 * d[0] - 1, c[1] + 2 * d[1] - 1, c[2] + 2 * d[2] - 1)
            c2 = cell_of(raw)
            if c2 not in Wc or c2 < c:
                continue
            n1, n2 = set(lab.values()), set(netc.get(c2, {}).values())
            kind = "same network" if n1 == n2 else "different networks"
            if raw != c2:
                kind += " (across torus wrap)"
            cen[kind] = cen.get(kind, 0) + 1
    out["contacts"] = cen
    out["t_build_and_typing_s"] = round(time.time() - t0, 2)

    # ---------------- my decider
    t1 = time.time()
    inst, sol = D.decide(L, Wbg, Wc)
    out["decider_planted"] = sol is not None
    out["n_components"], out["n_parity1_joins"] = int(inst.ncomp), int(inst.n_par1)
    out["affine_part_consistent"] = bool(inst.affine_ok)
    out["n_search_cells"] = len(inst.search_cells) if hasattr(inst, "search_cells") else None
    out["E0"] = inst.E0
    out["agree_with_NAE"] = out["decider_planted"] == out["nae_sat"]
    if sol is not None:
        sigma = inst.expand(sol)
        local_ok, Hc, He = inst.verify(sigma)
        out["returned_state_all_cells_local_min"] = local_ok
        out["returned_state_H_cellsum"], out["returned_state_H_edgesum"] = Hc, He
        out["returned_state_is_ground_state(H==E0)"] = (Hc == He == inst.E0)
        # read-off: x_i = sigma_v for v in V_{x_i}; check constancy and NAE
        assign, const = {}, True
        for x, vs in verts_by_net.items():
            vals = {int(sigma[(v[0] * L + v[1]) * L + v[2]]) for v in vs}
            const &= len(vals) == 1
            if x.startswith("x"):
                assign[int(x[1:])] = vals.pop()
        a = tuple(assign.get(i, 0) for i in range(n))
        out["sigma_constant_on_every_network"] = const
        out["read_off_assignment"] = a
        out["read_off_NAE_satisfies"] = a in set(sols)
    out["t_decide_s"] = round(time.time() - t1, 2)

    # ---------------- pinned assignments: pin one vertex per variable network
    t2 = time.time()
    if not hasattr(inst, "search_cells"):
        inst.prepare_search()
    occ = sorted({v for c in cl for v, _ in c})
    pin_vertex = {v: min(verts_by_net[f"x{v}"]) for v in occ}
    agree, total = 0, 0
    solset = set(sols)
    for a in itertools.product((0, 1), repeat=n):
        pins = {pin_vertex[v]: a[v] for v in occ}
        feas = inst.affine_ok and inst.search(pins) is not None
        total += 1
        agree += feas == (a in solset)
    out["pinned_agree"] = f"{agree}/{total}"
    out["t_pinned_s"] = round(time.time() - t2, 2)
    out["maxrss_MB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)
    out["t_total_s"] = round(time.time() - t0, 2)
    print(json.dumps(out, default=str))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"e2e_{name}_m{margin}.json"), "w") as fh:
        json.dump(out, fh, default=str, indent=1)
    return out


if __name__ == "__main__":
    main()
