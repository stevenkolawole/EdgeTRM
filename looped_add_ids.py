#!/usr/bin/env python3
"""Sidecar token ids for looped_decode.py v1 records (which stored text only), so that
analyze_looped.py can measure first divergence in tokens.  For every
<root>/<tag>/<decode>.jsonl whose records lack "ids", writes <decode>.ids.json =
{idx: ids} by re-encoding the stored text with the model's tokenizer (decode -> encode
round trip; exact for these BPE tokenizers on text without special tokens, which the
decoder already stripped).  Never modifies the jsonl files.

    PYTHONPATH=~/pylibs/tf4554 ~/venvs/kvcache/bin/python looped_add_ids.py ~/breadth/lm/results/q6 [--decodes greedy]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--decodes", default="greedy")
    a = ap.parse_args()
    import looped_decode as D
    from transformers import AutoTokenizer
    toks = {}
    for d in sorted(Path(a.root).iterdir()):
        m = re.match(r"(.+)_s\d+_[a-z0-9]+_(core|all)$", d.name)
        if not d.is_dir() or not m or m.group(1) not in D.MODELS:
            continue
        fam = D.family(m.group(1))
        if fam.model_id not in toks:
            toks[fam.model_id] = AutoTokenizer.from_pretrained(fam.model_id, trust_remote_code=True)
        tok = toks[fam.model_id]
        for dec in a.decodes.split(","):
            p = d / f"{dec}.jsonl"
            if not p.exists():
                continue
            recs = D.read_jsonl(p)
            if all("ids" in r for r in recs.values()):
                continue
            side = {i: tok.encode(r["text"], add_special_tokens=False) for i, r in recs.items() if "ids" not in r}
            json.dump({"source": "re-encoded text", "tokenizer": fam.model_id, "ids": side},
                      open(d / f"{dec}.ids.json", "w"))
            print(d.name, dec, len(side))


if __name__ == "__main__":
    main()
