"""Cell adjacency between templates in neighbouring blocks (all 26 offsets, all 30 x 30 template pairs)."""
import os
import ast, itertools, re
txt = open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "construction"), "templates_out.txt")).read()
T = {}
for m_ in re.finditer(r"in (\(\d, -?1\)) out (\(\d, -?1\)): (\d+) cells: (\[.*?\])\n", txt):
    T[ast.literal_eval(m_.group(1)), ast.literal_eval(m_.group(2))] = ast.literal_eval(m_.group(4))
faces_of = {k: {k[0], k[1]} for k in T}
res = {}
for d in itertools.product((-1, 0, 1), repeat=3):
    if d == (0, 0, 0):
        continue
    for k1, c1 in T.items():
        s1 = set(c1)
        for k2, c2 in T.items():
            sh = [tuple(x + 8 * y for x, y in zip(c, d)) for c in c2]
            n = sum(1 for a in c1 for b in sh if all(abs(x - y) == 1 for x, y in zip(a, b)))
            if not n:
                continue
            face_nbr = sum(abs(x) for x in d) == 1
            if face_nbr:
                i = next(j for j in range(3) if d[j]); f1, f2 = (i, d[i]), (i, -d[i])
                consecutive = f1 in faces_of[k1] and f2 in faces_of[k2]
            else:
                consecutive = False
            key = ("face" if face_nbr else "edge/corner", "both use the shared face" if consecutive else "not consecutive")
            res[key] = res.get(key, 0) + 1
print("template pairs in neighbouring blocks with at least one vertex-sharing cell pair:", res)
