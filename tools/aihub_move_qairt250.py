"""Step 3: set aside every chain record produced on a QAIRT 2.50 compile (rename, never delete), with an index.
Verifies each chain against the account jobs.json (from_compile in the 2.50 set) before renaming. No tokens printed."""
import json, pathlib, sys, time
H = pathlib.Path.home() / "aihub_artifacts" / "h1"
Q250 = {"jpvln8vz5": "acct2 sudoku_mlp w8a8", "jgjr89e1p": "acct2 sudoku_mlp w8a16", "jpe7nqk85": "acct2 sudoku_attn w8a8",
        "j5wlrkq4p": "acct2 maze w8a8", "jgddmkv6g": "acct3 maze w8a8rope16", "jg9z7r08p": "acct2 sudoku_attn w8a16",
        "jgzl06r45": "acct2 maze fp32"}
jobs = {a: json.load(open(H / f"acct_{a}" / "jobs.json"))["jobs"] for a in ("acct2", "acct3")}


def compiles_of(acct, task, prec, device):
    ids = {j.get("from_compile") for j in jobs[acct] if j["kind"] in ("inference", "profile") and not j.get("ctl")
           and j["task"] == task and j["prec"] == prec and j["device"] == device}
    ids.discard(None)
    if not ids:   # chain failed before any device job: the compile the account had for (task, prec)
        ids = {j["job_id"] for j in jobs[acct] if j["kind"] == "compile" and j["task"] == task and j["prec"] == prec
               and j["runtime"] == "qnn_dlc" and "qairt" not in j}
        return ids, "no device job recorded; account compile for the cell"
    return ids, "from_compile of the chain device jobs"


A2 = ["Samsung Galaxy S22 5G", "Samsung Galaxy S23", "Samsung Galaxy S25", "Samsung Galaxy Tab S8",
      "Snapdragon 8 Elite Gen 5 QRD", "Samsung Galaxy S21", "Snapdragon 7 Gen 4 QRD"]
A3 = ["Arduino VENTUNO Q", "Dragonwing IQ-9075 EVK", "QCS8550 (Proxy)", "SA8775P ADP", "Snapdragon X Elite CRD",
      "Dragonwing RB3 Gen 2 Vision Kit", "Dragonwing Q-6690 MTP"]
plan = [("acct2", t, p, d) for d in A2 for t, p in (("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a8"), ("maze", "w8a8"))]
plan += [("acct3", "maze", "w8a8rope16", d) for d in A3]
index = {"written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "why": "compile ran on QAIRT 2.50.0.260828221209 (hub default from Sep 26): z_L_out carry compiled with a zero-point-0 "
                "encoding, clipped to [0, ~1.3] on every step (reports/edgetrm_exp/DIAG_ACCT_SPLIT.md). Rerun on the 2.45 pin (aihub_acct.QAIRT).",
         "qairt250_compiles": Q250, "moved": [], "ctl": [], "stubs": [], "problems": []}
todo = []
for acct, task, prec, dev in plan:
    tag = f"{prec}_{dev.replace(' ', '_')}_{acct}"
    d = H / task / "infer"
    files = [f for f in (d / f"dev_{tag}.json", d / f"dev_{tag}.npz") if f.exists()]
    if not files:
        index["problems"].append(f"no record for {task} {tag}")
        continue
    ids, how = compiles_of(acct, task, prec, dev)
    meta = json.load(open(d / f"dev_{tag}.json"))
    ok = bool(ids) and ids <= set(Q250)
    row = {"task": task, "chain": tag, "files": [f.name for f in files], "compile_jobs": sorted(ids), "evidence": how,
           "state": "done" if meta.get("done") else ("failed: " + str(meta.get("failed"))[:120] if meta.get("failed") else "partial"),
           "steps": len(meta.get("steps", [])), "step8_exact": (meta["steps"][-1]["exact"] if meta.get("done") else None)}
    if not ok:
        index["problems"].append({"chain": tag, "task": task, "compile_jobs": sorted(ids), "note": "NOT all 2.50, left in place"})
        print("SKIP (not 2.50):", task, tag, ids)
        continue
    todo.append((files, row))
ctl = [("sudoku_mlp", "ctl_w8a8-w8a16_Samsung_Galaxy_S24_acct2", "cheap w8a8 jpvln8vz5 + precise w8a16 jgjr89e1p both 2.50; cheap steps and profiles recorded"),
       ("sudoku_attn", "ctl_w8a8-w8a16_Samsung_Galaxy_S24_acct2", "cheap w8a8 jpe7nqk85 + precise w8a16 jg9z7r08p both 2.50; nothing run"),
       ("maze", "ctl_w8a8rope16-fp32_Samsung_Galaxy_S25_acct3", "cheap w8a8rope16 jgddmkv6g 2.50 (precise fp32 jgzllkxx5 2.45)")]
for task, tag, why in ctl:
    d = H / task / "infer"
    files = [f for f in (d / f"{tag}.json", d / f"{tag}.npz") if f.exists()]
    meta = json.load(open(d / f"{tag}.json")) if (d / f"{tag}.json").exists() else {}
    index["ctl"].append({"task": task, "chain": tag, "files": [f.name for f in files], "why": why,
                         "steps": len(meta.get("steps", [])), "profile": {k: v.get("job_id") for k, v in meta.get("profile", {}).items()},
                         "current": meta.get("current")})
    todo.append((files, None))
stubs = [("sudoku_mlp", "ctl_w8a8-w8a16_Samsung_Galaxy_S25_acct2.json"), ("sudoku_mlp", "ctl_w8a8-w8a16_Snapdragon_8_Elite_QRD_acct3.json"),
         ("sudoku_attn", "ctl_w8a8-w8a16_Snapdragon_8_Elite_QRD_acct3.json"), ("maze", "ctl_w8a8rope16-fp32_Samsung_Galaxy_S25_acct2.json"),
         ("maze", "ctl_w8a8rope16-fp32_Snapdragon_8_Elite_QRD_acct3.json")]
if "--go" not in sys.argv:
    print("DRY RUN: would move", sum(len(f) for f, _ in todo), "files;", len(index["problems"]), "problems")
    for f, row in todo:
        print("  ", [str(x.relative_to(H)) for x in f], (row or {}).get("compile_jobs"), (row or {}).get("state"))
    print("problems:", json.dumps(index["problems"], indent=1))
    sys.exit(0)
for files, row in todo:
    for f in files:
        f.rename(f.with_name(f.name + ".qairt250"))
    if row:
        index["moved"].append(row)
for task, name in stubs:
    f = H / task / "infer" / name
    if f.exists():
        meta = json.load(open(f))
        if meta.get("steps") or meta.get("current") or meta.get("profile"):
            index["problems"].append(f"stub {task}/{name} is not empty, left in place")
            continue
        f.rename(f.with_name(name + ".stub_unhosted"))
        index["stubs"].append(f"{task}/{name} -> .stub_unhosted (launch-1 placeholder; device no longer hosted or moved to the other account)")
# acct2 S24 maze ctl record stays (cheap rope16 = 2.45 jgzlzmjz5); its fp32 profile ran on the 2.50 jgzl06r45 -> entry dropped so the pinned compile is re-profiled
f = H / "maze" / "infer" / "ctl_w8a8rope16-fp32_Samsung_Galaxy_S24_acct2.json"
meta = json.load(open(f))
pf = meta.get("profile", {}).get("fp32")
if pf:
    f.with_name(f.name + ".pre_qairt_fix").write_text(json.dumps(meta, indent=1))
    meta["profile"].pop("fp32")
    f.write_text(json.dumps(meta, indent=1))
    index["ctl"].append({"task": "maze", "chain": f.stem, "kept": True, "removed_profile": pf.get("job_id"),
                         "edit": "profile.fp32 entry removed (it ran on 2.50 jgzl06r45); json before the edit kept as " + f.name + ".pre_qairt_fix"})
(H / "QAIRT250_MOVED_INDEX.json").write_text(json.dumps(index, indent=1))
print("moved", len(index["moved"]), "dev chains,", len(index["ctl"]), "ctl entries,", len(index["stubs"]), "stubs; problems:", json.dumps(index["problems"]))
print("index:", H / "QAIRT250_MOVED_INDEX.json")
