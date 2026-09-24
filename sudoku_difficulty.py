#!/usr/bin/env python3
"""Model-independent Sudoku difficulty measures (X1), added to results/breadth/sudoku_extreme_test_meta.json:

  singles_residual  blanks left after naked + hidden singles are applied to a fixpoint (0 = the puzzle
                    yields to singles alone); tracks the fp32 models' iterations-to-solve better than the
                    dataset's tdoku backtrack rating (Spearman with fp32 stable solve step 0.54-0.58 vs
                    0.30-0.43 on TRM-MLP / TRM-attn / HRM)
  mrv_backtracks    dead ends of a depth-first search with singles propagation and minimum-remaining-
                    values branching (a second solver's backtrack count; the solution is checked against
                    the dataset's answer)

    python sudoku_difficulty.py /path/to/test.csv     # recomputes both measures for the meta file's puzzles
"""
import numpy as np
UNITS=[]
for r in range(9): UNITS.append([r*9+c for c in range(9)])
for c in range(9): UNITS.append([r*9+c for r in range(9)])
for br in range(3):
    for bc in range(3): UNITS.append([(br*3+r)*9+bc*3+c for r in range(3) for c in range(3)])
PEERS=[set() for _ in range(81)]
for u in UNITS:
    for i in u: PEERS[i].update(u)
for i in range(81): PEERS[i].discard(i)
PEERS=[sorted(p) for p in PEERS]
CELL_UNITS=[[u for u in UNITS if i in u] for i in range(81)]

def propagate(grid, hidden=True, pairs=False):
    """grid: list of 81 ints (0 blank). naked + hidden singles to fixpoint.
    returns (filled grid, remaining blanks, rounds, cand) or None if contradiction"""
    g=list(grid)
    cand=[0 if g[i] else 0x3FE for i in range(81)]
    for i in range(81):
        if g[i]:
            for p in PEERS[i]: cand[p]&=~(1<<g[i])
    rounds=0
    while True:
        changed=False; rounds+=1
        # naked singles
        for i in range(81):
            if g[i]==0:
                c=cand[i]
                if c==0: return None
                if c&(c-1)==0:
                    v=c.bit_length()-1; g[i]=v; cand[i]=0; changed=True
                    for p in PEERS[i]: cand[p]&=~(1<<v)
        if hidden:
            for u in UNITS:
                for v in range(1,10):
                    if any(g[i]==v for i in u): continue
                    pos=[i for i in u if g[i]==0 and cand[i]>>v&1]
                    if len(pos)==0: return None
                    if len(pos)==1:
                        i=pos[0]; g[i]=v; cand[i]=0; changed=True
                        for p in PEERS[i]: cand[p]&=~(1<<v)
        if not changed: break
    return g, sum(1 for x in g if x==0), rounds, cand

def search_count(grid):
    """backtracks of a DFS with singles propagation and MRV (minimum remaining values) branching."""
    nodes=[0]
    def rec(g):
        res=propagate(g)
        if res is None: nodes[0]+=1; return None
        g2,rem,_,cand=res
        if rem==0: return g2
        i=min((j for j in range(81) if g2[j]==0), key=lambda j: bin(cand[j]).count('1'))
        for v in range(1,10):
            if cand[i]>>v&1:
                g3=list(g2); g3[i]=v
                out=rec(g3)
                if out is not None: return out
        nodes[0]+=1
        return None
    sol=rec(list(grid))
    return nodes[0], sol

def parse(q):
    return [0 if ch=='.' else int(ch) for ch in q]


if __name__ == "__main__":
    import csv
    import json
    import sys
    from pathlib import Path
    meta_p = Path(__file__).resolve().parent / "results" / "breadth" / "sudoku_extreme_test_meta.json"
    meta = json.load(open(meta_p))
    rows = list(csv.reader(open(sys.argv[1])))[1:]
    for name, idx in (("strat", meta["strat"]["rows"]), ("first1000", list(range(len(meta["first1000"]["rating"]))))):
        res, bt = [], []
        for r in idx:
            q = parse(rows[r][1])
            res.append(propagate(q)[1])
            n, sol = search_count(q)
            assert "".join(map(str, sol)) == rows[r][2], f"row {r}: solver disagrees with the dataset answer"
            bt.append(n)
        meta[name]["singles_residual"] = res
        meta[name]["mrv_backtracks"] = bt
    json.dump(meta, open(meta_p, "w"))
    print("updated", meta_p)
