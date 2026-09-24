"""Shard-by-shard bf16 copy of a (fp32) safetensors checkpoint, for hosts with little disk:
download one shard, cast floating tensors to bfloat16 (torch rounding), save, delete the fp32
shard, next.  Peak extra disk = one fp32 shard.  Output: ~/hf_models/<name> with config,
tokenizer and remote-code files copied.  usage: convert_bf16_stream.py NAME REPO MIN_FREE_GB"""
import json, os, shutil, sys, time
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_file, save_file

name, repo, min_free = sys.argv[1], sys.argv[2], float(sys.argv[3])
tmp = Path.home() / "breadth/lm/tmp_dl" / name
out = Path.home() / "hf_models" / name
tmp.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
info = HfApi().model_info(repo)
files = [s.rfilename for s in info.siblings]
for f in files:
    if not f.endswith(".safetensors") and not f.startswith(".git"):
        p = hf_hub_download(repo, f, revision=info.sha, local_dir=tmp)
        shutil.copy(p, out / f)
idx = json.load(open(out / "model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
total = 0
for sh in shards:
    if (out / sh).exists():
        print("have", sh, flush=True)
        continue
    while shutil.disk_usage(str(Path.home())).free / 2**30 < min_free:
        print(f"waiting for {min_free} GB free", flush=True)
        time.sleep(120)
    p = hf_hub_download(repo, sh, revision=info.sha, local_dir=tmp)
    sd = load_file(p)
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    save_file(sd, str(out / (sh + ".part")), metadata={"format": "pt"})
    os.replace(out / (sh + ".part"), out / sh)
    total += sum(v.numel() * v.element_size() for v in sd.values())
    os.remove(p)
    print("converted", sh, flush=True)
idx["metadata"] = {"total_size": sum(os.path.getsize(out / s) for s in shards)}
json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=2)
cfg = json.load(open(out / "config.json"))
cfg["torch_dtype"] = "bfloat16"
json.dump(cfg, open(out / "config.json", "w"), indent=2)
open(out / "SOURCE.txt", "w").write(f"{repo} revision {info.sha}, cast shard-by-shard to bfloat16\n")
shutil.rmtree(tmp)
print("done", out, flush=True)
