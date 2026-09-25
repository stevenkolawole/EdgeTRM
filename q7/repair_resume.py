# Q7 (X5), Sep 25 04:10: repair base runs frozen by the upstream resume bug (load_state_dict(assign=True) after
# the optimizers were built). The raw weights and optimizer state in resume.pt are those of the resume step
# (never updated since); the EMA shadow drifted to the raw weights. Reset step/iter/batch to the resume
# point, set the EMA shadow = raw weights (deviation: EMA re-initialised there), move later checkpoints,
# their evals and fork files to invalid_resume_bug/.
import sys, os, shutil, torch, re
run, step, it, skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = os.path.expanduser(f"~/q7/ckpt/{run}")
p = torch.load(f"{d}/resume.pt", map_location="cpu", weights_only=False)
print(run, "resume.pt had step", p["step"], "iter", p["iter_id"], "batch", p["batch_count"])
msd = p["model_state_dict"]
ema = p.get("ema_state", {})
n = 0
for k in list(ema):
    kk = k if k in msd else None
    if kk is None:
        cand = [m for m in msd if m.endswith(k.split("_orig_mod.")[-1])]
        kk = cand[0] if len(cand) == 1 else None
    assert kk is not None, k
    ema[k] = msd[kk].clone().to(ema[k].device if hasattr(ema[k], "device") else "cpu"); n += 1
p.update(step=step, iter_id=it, batch_count=skip, dataset_iters=it, ema_state=ema)
os.makedirs(f"{d}/invalid_resume_bug", exist_ok=True)
for f in os.listdir(d):
    m = re.fullmatch(r"(?:eval_)?step_(\d+)(?:\.json|\.nolsq)?|resume_iter(\d+)\.pt|DONE", f)
    if not m: continue
    s = int(m.group(1)) if m.group(1) else (6510 * (int(m.group(2)) + 1) if m.group(2) else 10**9)
    if s > step:
        shutil.move(f"{d}/{f}", f"{d}/invalid_resume_bug/{f}"); print("  moved", f)
shutil.copy(f"{d}/resume.pt", f"{d}/invalid_resume_bug/resume.pt.frozen")
torch.save(p, f"{d}/resume.pt.tmp"); os.replace(f"{d}/resume.pt.tmp", f"{d}/resume.pt")
print("  repaired: step", step, "iter", it, "skip", skip, "ema tensors reset", n)
