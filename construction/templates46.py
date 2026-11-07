"""Sparse oriented block templates for the G46 profile (F6 wires): besides Lemma 4.3's conditions,
  (C1) no template cell lies in an extreme layer of the block except the two outer crossing cells
       (=> wires in different blocks never touch, except linked crossing pairs);
  (C2) every template cell touches at most one unlinked template cell (its 'contact'), so every F6 wire cell has
       at most one floating corner from its own template.
Search: depth-first with iterative deepening over T+ steps."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks as B  # noqa: E402
from blocks import H, TPLUS, vadd, vsub  # noqa: E402

STEPS = TPLUS


def extreme(c, C=(0, 0, 0)):
    return any(c[j] - C[j] in (-H, H - 1) for j in range(3))


def adjacent(a, b):
    return all(abs(x - y) == 1 for x, y in zip(a, b))


def contacts(path):
    cnt = {c: 0 for c in path}
    idx = {c: t for t, c in enumerate(path)}
    for c in path:
        for a in STEPS + [tuple(-x for x in s) for s in STEPS]:
            r = vadd(c, a)
            if r in idx and abs(idx[r] - idx[c]) > 1:
                cnt[c] += 1
    return cnt


def sparse_template(f1, f2, max_len=24):
    t1, t2 = B.entry_tau(f1), f2[1]
    o1, i1 = B.boundary_pair((0, 0, 0), f1, t1)
    o2, i2 = B.boundary_pair((0, 0, 0), f2, t2)
    a_out = vsub(o2, i2)
    if f1[0] == f2[0]:
        return B.make_template(f1, f2)

    def dist(c):
        return max(abs(x - y) for x, y in zip(c, i2))

    best = None
    for limit in range(4, max_len):
        path = [o1, i1]
        cnt = {o1: 0, i1: 0}
        result = []

        def ok_add(n):
            if n in cnt or not B.in_block((0, 0, 0), n) or (extreme(n) and n != o2):
                return None
            touched = [r for r in cnt if adjacent(r, n) and r != path[-1]]
            if len(touched) > 1:
                return None
            if touched and cnt[touched[0]] >= 1:
                return None
            return touched

        def dfs(last):
            c = path[-1]
            if c == i2:
                if a_out == last or B.dot(last, a_out) != -1:
                    return False
                t = ok_add(o2)
                if t is None:
                    return False
                result.append(path + [o2])
                return True
            if len(path) >= limit or dist(c) > (limit - len(path)) + 1:
                return False
            for a in sorted((s for s in STEPS if s != last), key=lambda s: dist(vadd(c, s))):
                n = vadd(c, a)
                if n == o2:
                    continue
                t = ok_add(n)
                if t is None:
                    continue
                path.append(n)
                cnt[n] = len(t)
                for r in t:
                    cnt[r] += 1
                if dfs(a):
                    return True
                for r in t:
                    cnt[r] -= 1
                del cnt[n]
                path.pop()
            return False

        if dfs(vsub(i1, o1)):
            best = result[0]
            break
    return best


def check_sparse(f1, f2, cells):
    B.check_template(f1, f2, cells)
    assert all(not extreme(c) for c in cells[1:-1]), "C1"
    cnt = contacts(cells)
    assert max(cnt.values()) <= 1, ("C2", cnt)
    return sum(cnt.values()) // 2


TEMPLATES46 = {}


def templates46():
    if not TEMPLATES46:
        for f1 in B.FACES6:
            for f2 in B.FACES6:
                if f1 == f2:
                    continue
                t = sparse_template(f1, f2)
                assert t is not None, ("no sparse template", f1, f2)
                check_sparse(f1, f2, t)
                TEMPLATES46[f1, f2] = t
    return TEMPLATES46


if __name__ == "__main__":
    import time
    t0 = time.time()
    T = templates46()
    print(f"S={B.S}: 30 sparse oriented templates (C1 no extreme-layer cells except outer crossing cells;"
          f" C2 <= 1 contact per cell) found and checked:")
    for (f1, f2), cells in sorted(T.items()):
        print(f"  in {f1} out {f2}: {len(cells)} cells, {check_sparse(f1, f2, cells)} contact pairs: {cells}")
    print(f"[{time.time()-t0:.1f}s]")
