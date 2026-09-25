"""M1 per-puzzle decomposition of a hybrid c(T-k)+P(k): kept = fraction of the cheap map's solved puzzles still solved
after the polish; fixed = fraction of its unsolved puzzles the polish solves; cheap-only / fp32-only against fp32@T;
union = oracle of the two single runs. With residual records (c_res): AUROC of the cheap map's relative residual
||z_t - z_{t-1}|| / ||z_t|| at the polish start for "fixed" among the unsolved (0.5 = no relation).
    python m1_decomp.py 8 'results/breadth/babel/m1/p_*_T*_*.json'"""
import json, sys, glob, numpy as np
def arr(s): return (np.frombuffer(s.encode(),np.uint8)-48).astype(int)
k=int(sys.argv[1]) if len(sys.argv)>1 else 8
pat=sys.argv[2] if len(sys.argv)>2 else "results/breadth/babel/m1/p_*_T*_*.json"
def auroc(score, lab):
    score, lab = np.asarray(score, float), np.asarray(lab, bool)
    if lab.all() or (~lab).all(): return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())
print("| file | cell | polish | fp32@T | cheap@T | cheap@(T-k) | union(fp32@T, cheap@T) | hybrid c(T-k)+P(k) | cheap-solved kept | cheap-unsolved fixed | cheap-only frac (vs fp32@T) | fp32-only frac | residual AUROC fixed vs not | median residual fixed / not |")
print("|"+"---|"*14)
for f in sorted(glob.glob(pat)):
    b=json.load(open(f)); T=b["T"]; fT=arr(b["fp32"]["at"][str(T)])
    for r in b["records"]:
        if "c_at" not in r or f"c{T-k}.P{k}" not in r["schedules"]: continue
        C=[arr(x) for x in r["c_at"]]; cT=C[T-1]; ck=C[T-k-1]
        h=arr(r["schedules"][f"c{T-k}.P{k}"]["correct"])
        kept=h[ck==1].mean() if ck.any() else float('nan'); fixed=h[ck==0].mean()
        au, med = "", ""
        cr = r.get("c_res", {}).get(str(T-k))
        if cr is not None:
            res = np.asarray(cr); un = ck == 0; fx = (h == 1) & un
            au = f"{auroc(res[un], fx[un]):.2f}"
            med = f"{np.median(res[fx]) if fx.any() else float('nan'):.3f} / {np.median(res[un & ~fx]):.3f}"
        print(f"| {f.split('/')[-1][2:-5]} | {r['cell']}#{r['draw']} | {r['polish']} | {fT.mean()*100:.1f} | {cT.mean()*100:.1f} | {ck.mean()*100:.1f} | {100*np.maximum(fT,cT).mean():.1f} | {100*h.mean():.1f} | {100*kept:.1f} | {100*fixed:.1f} | {100*np.mean(cT&(1-fT)):.1f} | {100*np.mean(fT&(1-cT)):.1f} | {au} | {med} |")
