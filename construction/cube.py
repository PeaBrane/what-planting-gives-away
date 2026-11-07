"""Unit-cube (cell) primitives for Hamze et al. tile planting.

Corner k <-> d = (dx, dy, dz) with k = dx + 2 dy + 4 dz.  A state is an 8-bit mask m with bit k set iff s_k = -1.
H_C(s) = - sum_{(a,b) in cube edges} W_ab s_a s_b, couplings W in {+1,-1}^12 ordered as CUBE_E.
"""
import itertools

CORNERS = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
CUBE_E = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
FACES = []  # (axis, value) -> corner set
for ax in range(3):
    for val in (0, 1):
        FACES.append(frozenset(k for k in range(8) if CORNERS[k][ax] == val))
FACE_EDGES = [[i for i, (a, b) in enumerate(CUBE_E) if a in f and b in f] for f in FACES]
FULL = 0xFF


def spin(m, k):
    return -1 if (m >> k) & 1 else 1


def energy(W, m):
    return -sum(w * spin(m, a) * spin(m, b) for w, (a, b) in zip(W, CUBE_E))


def gs_set(W):
    en = [energy(W, m) for m in range(256)]
    e0 = min(en)
    return frozenset(m for m in range(256) if en[m] == e0), e0


def frustrated_faces(W):
    out = []
    for fi, fe in enumerate(FACE_EDGES):
        p = 1
        for i in fe:
            p *= W[i]
        if p < 0:
            out.append(fi)
    return tuple(out)


def class_name(W):
    fr = frustrated_faces(W)
    n = len(fr)
    if n == 0:
        return "F0"
    if n == 6:
        return "F6"
    # opposite faces are (2a, 2a+1)
    if n == 2:
        a, b = fr
        return "F22" if a // 2 == b // 2 else "F21"
    un = [f for f in range(6) if f not in fr]
    a, b = un
    return "F42" if a // 2 == b // 2 else "F41"


def all_couplings():
    return [tuple(s) for s in itertools.product([1, -1], repeat=12)]


def gauge_W(W, t):
    """t: 8-bit mask of gauge flips; W'_ab = W_ab t_a t_b."""
    return tuple(w * spin(t, a) * spin(t, b) for w, (a, b) in zip(W, CUBE_E))


# octahedral group as corner permutations: coordinate permutation + reflections
def oh_perms():
    perms = []
    for p in itertools.permutations(range(3)):
        for r in itertools.product([0, 1], repeat=3):
            perm = []
            for k in range(8):
                d = CORNERS[k]
                nd = [d[p[i]] ^ r[i] for i in range(3)]
                perm.append(nd[0] + 2 * nd[1] + 4 * nd[2])
            perms.append(tuple(perm))
    return perms


def permute_W(W, perm):
    """Return W' with W'(perm[a], perm[b]) = W(a, b)."""
    idx = {e: i for i, e in enumerate(CUBE_E)}
    out = [0] * 12
    for i, (a, b) in enumerate(CUBE_E):
        pa, pb = perm[a], perm[b]
        j = idx[(min(pa, pb), max(pa, pb))]
        out[j] = W[i]
    return tuple(out)


def coupling_from_class(name, orient_perm=None, gauge=0):
    """Canonical representative of each class in ferromagnetic gauge (+1 is a ground state)."""
    W = [1] * 12
    idx = {e: i for i, e in enumerate(CUBE_E)}
    if name == "F0":
        pass
    elif name == "F21":
        W[idx[(0, 1)]] = -1          # one AFM edge: faces y0 and z0 frustrated
    elif name == "F22":
        W[idx[(0, 1)]] = -1          # edges (000-100) and (001-101)?  no: choose two parallel x-edges on face y=0
        W[idx[(4, 5)]] = -1          # (0,0,1)-(1,0,1): frustrates faces z0 and z1 only
    elif name == "F41":
        # four frustrated faces x0, y0, z0, z1 ; unfrustrated x1, y1 (adjacent)
        pass
    elif name == "F42":
        pass
    elif name == "F6":
        pass
    W = tuple(W)
    if name in ("F41", "F42", "F6"):
        target = {"F41": {0, 2, 4, 5}, "F42": {0, 1, 2, 3}, "F6": {0, 1, 2, 3, 4, 5}}[name]
        best = None
        for Wc in all_couplings():
            if set(frustrated_faces(Wc)) == target:
                G, _ = gs_set(Wc)
                if 0 in G and (best is None or sum(x < 0 for x in Wc) < sum(x < 0 for x in best)):
                    best = Wc
        W = best
    if orient_perm is not None:
        W = permute_W(W, orient_perm)
    if gauge:
        W = gauge_W(W, gauge)
    return W
