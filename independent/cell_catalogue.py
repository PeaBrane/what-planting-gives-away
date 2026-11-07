"""Independent catalogue of all 4096 +-1 cube couplings (imports nothing from construction/).

Conventions (as in construction/cube.py): corner k = dx + 2 dy + 4 dz; edges are pairs (a, b), a < b,
a ^ b a power of two, in lexicographic order; a state is an 8-bit mask, bit k set iff s_k = -1;
H_C = - sum_e W_e s_a s_b.
Checks: class table (Lemma 2.2), T-join duality (Lemma 2.1), clause cell (Lemma 2.3), affinity (Theorem B),
loop cone K(Q3) and MET(Q3) = CUT(Q3) (Corollaries 2(b) and 4).
"""
import os
import itertools
import sys

import numpy as np
from scipy.optimize import linprog

CORN = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
assert EDGES == [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
FACES = [(ax, v) for ax in range(3) for v in (0, 1)]          # face index f = 2 ax + v
FACE_CORNERS = [frozenset(k for k in range(8) if CORN[k][ax] == v) for ax, v in FACES]
FACE_EDGES = [[i for i, (a, b) in enumerate(EDGES) if a in F and b in F] for F in FACE_CORNERS]
EDGE_FACES = [[f for f in range(6) if i in FACE_EDGES[f]] for i in range(12)]   # dual edge of cube edge i
FULL = 255

SP = np.array([[(-1 if (m >> k) & 1 else 1) for k in range(8)] for m in range(256)])
PR = np.stack([SP[:, a] * SP[:, b] for a, b in EDGES], axis=1)          # (256, 12)
ALLW = np.array(list(itertools.product([1, -1], repeat=12)))            # (4096, 12)
EN = -(PR @ ALLW.T)                                                      # (256, 4096)


def gamma(j):
    col = EN[:, j]
    return frozenset(np.nonzero(col == col.min())[0].tolist()), int(col.min())


def frustrated(W):
    return tuple(f for f in range(6) if np.prod([W[i] for i in FACE_EDGES[f]]) < 0)


def hamze_class(T):
    n = len(T)
    if n in (0, 6):
        return f"F{n}"
    if n == 2:
        a, b = T
        return "F22" if FACES[a][0] == FACES[b][0] else "F21"
    un = [f for f in range(6) if f not in T]
    return "F42" if FACES[un[0]][0] == FACES[un[1]][0] else "F41"


def is_affine(G):
    g0 = next(iter(G))
    V = {m ^ g0 for m in G}
    return all((a ^ b) in V for a in V for b in V)


def mask(ks):
    return sum(1 << k for k in ks)


DOMINOES = {mask([a, b]) for a, b in EDGES}
FACEM = {mask(F) for F in FACE_CORNERS}


def min_tjoins(T):
    """Minimum T-joins of the octahedron (dual edges indexed like cube edges)."""
    best, out = None, []
    Ts = set(T)
    for J in range(1 << 12):
        deg = [0] * 6
        for i in range(12):
            if (J >> i) & 1:
                for f in EDGE_FACES[i]:
                    deg[f] += 1
        if {f for f in range(6) if deg[f] % 2} != Ts:
            continue
        w = bin(J).count("1")
        if best is None or w < best:
            best, out = w, [J]
        elif w == best:
            out.append(J)
    return best, out


def unsat_set(W, m):
    return sum(1 << i for i, (a, b) in enumerate(EDGES)
               if W[i] * (-1 if (m >> a) & 1 else 1) * (-1 if (m >> b) & 1 else 1) < 0)


def main():
    log = []

    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    stats = {}
    tj_cache = {}
    total_gs = 0
    planted = []
    lemma21_ok = True
    for j in range(4096):
        W = tuple(int(x) for x in ALLW[j])
        G, e0 = gamma(j)
        total_gs += len(G)
        if 0 in G:
            planted.append(W)
        T = frustrated(W)
        cls = hamze_class(T)
        g0 = min(G)
        V = frozenset(m ^ g0 for m in G)
        aff = is_affine(G)
        # Lemma 2.1: U maps Gamma 2-to-1 onto min T-joins; H = -12 + 2|U|
        if T not in tj_cache:
            tj_cache[T] = min_tjoins(T)
        w, joins = tj_cache[T]
        Us = {unsat_set(W, m) for m in G}
        if not (Us == set(joins) and len(G) == 2 * len(joins) and e0 == -12 + 2 * w):
            lemma21_ok = False
        # structure in the gauge with 0 in Gamma
        shape = None
        if len(V) == 2:
            shape = "rigid" if V == {0, FULL} else "?"
        elif len(V) == 4:
            nontriv = sorted(V - {0, FULL})
            D = nontriv[0]
            shape = "domino" if (D in DOMINOES or (FULL ^ D) in DOMINOES) else (
                "face" if D in FACEM else f"other{D}")
        else:
            shape = f"size{len(V)}"
        key = (cls, len(G), shape, aff)
        stats[key] = stats.get(key, 0) + 1
    p("Lemma 2.1 (T-join duality, H = -12 + 2|U|, 2-to-1 onto min T-joins) for all 4096:", lemma21_ok)
    p("class, |Gamma|, shape in a gauge with 0 in Gamma, affine  ->  #couplings")
    for k in sorted(stats):
        p("  ", k, stats[k])
    p("sum_W |Gamma(W)| =", total_gs, "; /256 =", total_gs / 256, "; #W with 0 in Gamma =", len(planted))

    # F22 columns: locked pairs are the edges perpendicular to the two frustrated faces
    f22_ok = True
    for j in range(4096):
        W = tuple(int(x) for x in ALLW[j])
        T = frustrated(W)
        if hamze_class(T) != "F22":
            continue
        G, _ = gamma(j)
        ax = FACES[T[0]][0]
        perp_edges = [(a, b) for a, b in EDGES if (a ^ b) == (1 << ax)]
        locked = [(a, b) for a, b in EDGES if len({((m >> a) ^ (m >> b)) & 1 for m in G}) == 1]
        if sorted(locked) != sorted(perp_edges):
            f22_ok = False
    p("F22: locked corner pairs are exactly the 4 edges perpendicular to the frustrated faces:", f22_ok)

    # Theorem B: <= 2 ground-state pairs -> affine; which classes
    tb = {}
    for j in range(4096):
        W = tuple(int(x) for x in ALLW[j])
        G, _ = gamma(j)
        cls = hamze_class(frustrated(W))
        tb.setdefault((cls, len(G) <= 4), set()).add(is_affine(G))
    p("Theorem B: (class, |Gamma|<=4) -> set of affine flags:", {k: sorted(v) for k, v in sorted(tb.items())})

    # Lemma 2.3: clause cell
    idx = {e: i for i, e in enumerate(EDGES)}
    W0 = [1] * 12
    W0[idx[(0, 1)]] = -1
    W0[idx[(4, 5)]] = -1
    W0 = tuple(W0)
    j0 = int(np.nonzero((ALLW == np.array(W0)).all(axis=1))[0][0])
    G0, _ = gamma(j0)
    p("W0 class:", hamze_class(frustrated(W0)), "|Gamma| =", len(G0))
    COLS = {"00": (0, 4), "10": (1, 5), "11": (3, 7), "01": (2, 6)}
    order = ["00", "10", "11", "01"]

    def colbits(m):
        bits = []
        for c in order:
            a, b = COLS[c]
            if ((m >> a) & 1) != ((m >> b) & 1):
                return None
            bits.append((m >> a) & 1)
        return tuple(bits)

    cb = sorted(colbits(m) for m in G0)
    p("Lemma 2.3(1) column-locked:", all(c is not None for c in cb))
    p("Lemma 2.3(2) column bits (t00,t10,t11,t01):", ["".join(map(str, c)) for c in cb])
    proj = sorted({c[1:] for c in cb})
    nae = lambda t: len(set(t)) > 1
    want3 = sorted(t for t in itertools.product((0, 1), repeat=3) if nae((t[0], 1 - t[1], t[2])))
    p("Lemma 2.3(3) projection:", proj, "== NAE(t10, not t11, t01):", proj == want3)

    def gauge(W, t):
        return tuple(w * (-1 if (t >> a) & 1 else 1) * (-1 if (t >> b) & 1 else 1) for w, (a, b) in zip(W, EDGES))

    item4 = True
    for eps in itertools.product((0, 1), repeat=3):
        t = 0
        if eps[0] == 1:
            t |= mask(COLS["10"])
        if eps[1] == 0:
            t |= mask(COLS["11"])
        if eps[2] == 1:
            t |= mask(COLS["01"])
        We = gauge(W0, t)
        je = int(np.nonzero((ALLW == np.array(We)).all(axis=1))[0][0])
        Ge, _ = gamma(je)
        cbe = [colbits(m) for m in Ge]
        if any(c is None for c in cbe) or hamze_class(frustrated(We)) != "F22":
            item4 = False
            continue
        pr = sorted({c[1:] for c in cbe})
        want = sorted(t3 for t3 in itertools.product((0, 1), repeat=3)
                      if nae((t3[0] ^ eps[0], t3[1] ^ eps[1], t3[2] ^ eps[2])))
        if pr != want:
            item4 = False
        p("  eps", eps, "gauge mask", t, "proj == NAE(t xor eps):", pr == want)
    p("Lemma 2.3(4) all 8 sign patterns:", item4)

    # Corollary 4: K(Q3) membership of all planted +-1 couplings, via the cycles of Q3
    cycles = []
    for J in range(1, 1 << 12):
        es = [EDGES[i] for i in range(12) if (J >> i) & 1]
        deg = {}
        for a, b in es:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        if any(d != 2 for d in deg.values()):
            continue
        # connected?
        vs = list(deg)
        seen, st = {vs[0]}, [vs[0]]
        while st:
            x = st.pop()
            for a, b in es:
                for u, v in ((a, b), (b, a)):
                    if u == x and v not in seen:
                        seen.add(v)
                        st.append(v)
        if len(seen) == len(vs):
            cycles.append(J)
    lens = sorted(bin(J).count("1") for J in cycles)
    p("cycles of Q3:", len(cycles), "by length:", {l: lens.count(l) for l in sorted(set(lens))})
    atoms = []
    for J in cycles:
        Cset = [i for i in range(12) if (J >> i) & 1]
        for f in Cset:
            v = np.zeros(12)
            v[Cset] = 1
            v[f] = -1
            atoms.append(v)
    A = np.array(atoms).T                                   # (12, #atoms)
    p("loop atoms:", A.shape[1])
    Aeq = np.hstack([A, np.eye(12)])
    nK = 0
    for W in planted:
        r = linprog(np.zeros(Aeq.shape[1]), A_eq=Aeq, b_eq=np.array(W, float), bounds=(0, None), method="highs")
        nK += r.status == 0
    p("planted +-1 couplings in K(Q3):", nK, "/", len(planted))
    # random real W on the boundary of P(Q3): check membership in K(Q3) (Seymour for Q3, real weights)
    rng = np.random.default_rng(12345)
    cuts = []
    for S in range(1, 128):         # S not containing corner 7 suffices (delta(S) = delta(S^c))
        cuts.append([1.0 if (((S >> a) & 1) != ((S >> b) & 1)) else 0.0 for a, b in EDGES])
    Dm = np.array(cuts)
    nreal, nrealK = 0, 0
    for _ in range(400):
        W = rng.normal(size=12)
        lam = max(0.0, np.max(-(Dm @ W) / Dm.sum(axis=1)))
        W = W + lam * np.ones(12)
        assert (Dm @ W).min() > -1e-9
        nreal += 1
        r = linprog(np.zeros(Aeq.shape[1]), A_eq=Aeq, b_eq=W, bounds=(0, None), method="highs")
        nrealK += r.status == 0
    p("random real W in P(Q3) (on its boundary) that lie in K(Q3):", nrealK, "/", nreal)
    # and the converse inclusion K(Q3) subset P(Q3): every atom has atom . delta(S) >= 0
    p("all loop atoms in P(Q3):", bool((Dm @ A).min() >= 0))

    # Corollary 2(b) ingredient: MET(Q3) = CUT(Q3) on all +-1 objectives
    rows, rhs = [], []
    for J in cycles:
        Cset = [i for i in range(12) if (J >> i) & 1]
        for r_ in range(1, len(Cset) + 1, 2):
            for F in itertools.combinations(Cset, r_):
                v = np.zeros(12)
                v[Cset] = -1
                v[list(F)] = 1
                rows.append(v)
                rhs.append(len(F) - 1)
    Aub, bub = np.array(rows), np.array(rhs, float)
    p("MET(Q3) cycle inequalities:", Aub.shape[0])
    bad = 0
    for j in range(4096):
        W = ALLW[j].astype(float)
        r = linprog(2 * W, A_ub=Aub, b_ub=bub, bounds=(0, 1), method="highs")
        lp = -W.sum() + r.fun
        e0 = EN[:, j].min()
        if abs(lp - e0) > 1e-7:
            bad += 1
    p("+-1 couplings with min_{MET(Q3)} != min H_C:", bad, "/ 4096")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell_catalogue_out.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
