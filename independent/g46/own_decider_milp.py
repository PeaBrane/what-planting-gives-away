"""Cross-check of the DPLL search: the same reduced table constraints (own_decider.prepare_search) solved as a
0/1 MILP with scipy HiGHS (no symmetry breaking, no propagation of ours).  Run: python -I own_decider_milp.py INST"""
import os
import json
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import own_decider as D  # noqa: E402

t0 = time.time()
z = np.load(sys.argv[1])
meta = json.loads(str(z["meta"]))
inst = D.Instance(int(z["L"]), z["types"], z["tcell"])
assert inst.solve_affine()
inst.prepare_search()
nv = len(inst.var_id)
rows, cols, vals, lo, hi = [], [], [], [], []
r = 0
ncol = nv
for c, refs, masks in inst.cells:
    ys = list(range(ncol, ncol + len(masks)))
    ncol += len(masks)
    for y in ys:
        rows.append(r); cols.append(y); vals.append(1.0)
    lo.append(1.0); hi.append(1.0); r += 1
    for k, (v, p) in enumerate(refs):
        rows.append(r); cols.append(v); vals.append(1.0 if p else -1.0)
        for y, m in zip(ys, masks):
            if (m >> k) & 1:
                rows.append(r); cols.append(y); vals.append(1.0)
        rhs = 1.0 if p else 0.0
        lo.append(rhs); hi.append(rhs); r += 1
A = coo_matrix((vals, (rows, cols)), shape=(r, ncol)).tocsr()
res = milp(np.zeros(ncol), constraints=LinearConstraint(A, lo, hi), integrality=np.ones(ncol),
           bounds=Bounds(np.zeros(ncol), np.ones(ncol)), options={"time_limit": 70, "presolve": True})
print(f"[{meta['name']}] MILP over {nv} variables, {len(inst.cells)} table constraints ({ncol} columns, {r} rows):"
      f" status={res.status} ({res.message}); feasible={res.status == 0}  [{time.time()-t0:.1f}s RSS {D.rss():.0f}MB]")
