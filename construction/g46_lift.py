"""Lift lemma for the G46 foam at S = 12: in the NON-periodic 3x3x3-block box of foam (extreme-layer cells of every
block minus all possible crossing cells), pick reference vertices v in the interface slab between the centre block
and a neighbour that lie in the giant component of the periodic 4x4x4 foam; check v ~ v + S e_j (j = x, y, z)
inside the box.  If so, the giant component of the infinite periodic foam is connected (its translates by the
period lattice S Z^3 are all the same component)."""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
from g46_local import DUF, plate_locks  # noqa: E402
from g46_local2 import L1  # noqa: E402
from g46_local3 import block_foam, comps, corner, lock  # noqa: E402

S = B.S
t0 = time.time()
r1, (P, cells, uf_per, big) = L1()
giant = big[0]
# foam of the non-periodic box: blocks -1..1 in each axis, plus one more layer so the box's own outer interfaces
# are complete slabs (blocks -2..2); reference vertices are taken near the centre
box = list(itertools.product(range(-2, 3), repeat=3))
foam = []
for g in box:
    foam += block_foam(g)
uf = DUF()
lock(uf, foam)
cnt = comps(uf, foam)
# reference vertices: corners of foam cells of the centre block that are in the periodic giant component
centre = block_foam((0, 0, 0))
refs = []
for u in centre:
    for k in range(8):
        v = corner(u, k)
        vp = tuple(x % p for x, p in zip(v, P))
        if uf_per.find(vp) == giant:
            refs.append(v)
res = {}
for j in range(3):
    e = tuple(S if i == j else 0 for i in range(3))
    joined = sum(1 for v in refs if uf.find(v) == uf.find(B.vadd(v, e)))
    res["xyz"[j]] = f"{joined}/{len(refs)} refs joined to their translate"
same = len({uf.find(v) for v in refs})
print(f"periodic 4x4x4 foam: {r1}")
print(f"non-periodic 5x5x5-block box: {len(foam)} foam cells, {len(cnt)} components;"
      f" centre-block refs in periodic giant: {len(refs)}, distinct box components among them: {same}; {res}"
      f"  [{time.time()-t0:.1f}s]")
