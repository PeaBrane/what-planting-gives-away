"""Independent cube primitives, written from the definitions (construction notes Sec. 0, 2.2, 8.2); imports nothing construction.

Corner k = dx + 2dy + 4dz is vertex u + d.  Edges: pairs (a, b), a < b, a ^ b a power of two, ascending.
H_C(s) = -sum W_ab s_a s_b; state bit 1 means s = -1.
"""
import itertools

import numpy as np

CORN = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
FULL = 255
FACE_SETS = [frozenset(k for k in range(8) if CORN[k][ax] == v) for ax in range(3) for v in (0, 1)]
FACE_EDGE_IDX = [[i for i, (a, b) in enumerate(EDGES) if a in f and b in f] for f in FACE_SETS]

_SPIN = np.array([[(-1 if (m >> k) & 1 else 1) for k in range(8)] for m in range(256)], dtype=np.int64)
_PROD = np.stack([_SPIN[:, a] * _SPIN[:, b] for a, b in EDGES], axis=1)  # (256, 12)


def energies(W):
    return -(_PROD @ np.asarray(W, dtype=np.int64))


def gamma(W):
    e = energies(W)
    return frozenset(np.nonzero(e == e.min())[0].tolist())


def n_frustrated(W):
    return sum(1 for fe in FACE_EDGE_IDX if np.prod([W[i] for i in fe]) < 0)


def cls(W):
    """Hamze class from the frustrated faces (faces 2a, 2a+1 are opposite)."""
    fr = [fi for fi, fe in enumerate(FACE_EDGE_IDX) if np.prod([W[i] for i in fe]) < 0]
    n = len(fr)
    if n in (0, 6):
        return f"F{n}"
    if n == 2:
        return "F22" if fr[0] // 2 == fr[1] // 2 else "F21"
    un = [f for f in range(6) if f not in fr]
    return "F42" if un[0] // 2 == un[1] // 2 else "F41"


def mask(ks):
    m = 0
    for k in ks:
        m |= 1 << k
    return m


def lock_pairs(G):
    """Lock classes: corners whose xor is constant over G.  Returns (classes, parity of each corner to its class
    representative, affine flag = |G| == 2^#classes)."""
    G = sorted(G)
    reps, cl = [], [None] * 8
    for k in range(8):
        for ci, r in enumerate(reps):
            if len({((m >> k) ^ (m >> r)) & 1 for m in G}) == 1:
                cl[k] = ci
                break
        else:
            cl[k] = len(reps)
            reps.append(k)
    par = [((G[0] >> k) ^ (G[0] >> reps[cl[k]])) & 1 for k in range(8)]
    return cl, par, len(G) == 2 ** len(reps)


ALL_W = [tuple(w) for w in itertools.product([1, -1], repeat=12)]
_G_ALL = None


def all_gammas():
    global _G_ALL
    if _G_ALL is None:
        E = -(_PROD @ np.array(ALL_W, dtype=np.int64).T)  # (256, 4096)
        mn = E.min(axis=0)
        _G_ALL = [frozenset(np.nonzero(E[:, j] == mn[j])[0].tolist()) for j in range(E.shape[1])]
    return _G_ALL


_F6 = None


def f6_gammas():
    global _F6
    if _F6 is None:
        _F6 = [(W, G) for W, G in zip(ALL_W, all_gammas()) if cls(W) == "F6"]
    return _F6


_PIN = {}


def pinnable(D, P):
    """Exists an F6 coupling whose ground states include 0 and 1_D and whose ground states vanishing on the
    corner set P are exactly {0, 1_D}.  (Lemma 8.1's condition, with the gauge free.)"""
    key = (D, P)
    if key not in _PIN:
        _PIN[key] = any(0 in G and D in G and {m for m in G if m & P == 0} == {0, D} for _, G in f6_gammas())
    return _PIN[key]


DOMINOES = [mask((a, b)) for a, b in EDGES]
