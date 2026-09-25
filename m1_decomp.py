import json, sys, glob, numpy as np
def arr(s): return (np.frombuffer(s.encode(),np.uint8)-48).astype(int)
k=int(sys.argv[1]) if len(sys.argv)>1 else 8
print("| file | cell | polish | fp32@T | cheap@T | cheap@(T-k) | union(fp32@T, cheap@T) | hybrid c(T-k)+P(k) | cheap-solved kept | cheap-unsolved fixed | cheap-only frac (vs fp32@T) | fp32-only frac |")
print("|"+"---|"*12)
for f in sorted(glob.glob("results/breadth/babel/m1/p_*_T*_*.json")):
    b=json.load(open(f)); T=b["T"]; fT=arr(b["fp32"]["at"][str(T)])
    for r in b["records"]:
        if "c_at" not in r or f"c{T-k}.P{k}" not in r["schedules"]: continue
        C=[arr(x) for x in r["c_at"]]; cT=C[T-1]; ck=C[T-k-1]
        h=arr(r["schedules"][f"c{T-k}.P{k}"]["correct"])
        kept=h[ck==1].mean() if ck.any() else float('nan'); fixed=h[ck==0].mean()
        print(f"| {f.split('/')[-1][2:-5]} | {r['cell']}#{r['draw']} | {r['polish']} | {fT.mean()*100:.1f} | {cT.mean()*100:.1f} | {ck.mean()*100:.1f} | {100*np.maximum(fT,cT).mean():.1f} | {100*h.mean():.1f} | {100*kept:.1f} | {100*fixed:.1f} | {100*np.mean(cT&(1-fT)):.1f} | {100*np.mean(fT&(1-cT)):.1f} |")
