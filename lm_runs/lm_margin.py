"""bf16 teacher-forced answer-logit margin per looped LM, from looped_decode v2 reference files
(<dir>/ref/<model>_s<steps>_n<n>_v2.pt: top-64 log-probs at every gold-solution position) and the matching
bf16 tf2 records (answer-token count).  margin = log p(top1) - log p(top2), mean over (a) all gold positions,
(b) the gold final-answer tokens (the positions before the end marker).  Label-free w.r.t. the quantized model.
usage: python lm_margin.py <ref.pt> <tf2.json of the bf16 cell> -> prints JSON"""
import json, sys
import torch
ref = torch.load(sys.argv[1])
recs = json.load(open(sys.argv[2]))["records"]
allm, ansm = [], []
for i, tl in enumerate(ref["top_lp"]):
    tl = tl.float()
    m = (tl[:, 0] - tl[:, 1])
    allm.append(float(m.mean()))
    r = recs[i]
    k = r.get("tf_ans_tokens", 0)
    n = m.shape[0]
    if k:
        ansm.append(float(m[n - 1 - k:n - 1].mean()))
print(json.dumps({"ref": sys.argv[1], "n": len(allm), "margin_all": sum(allm) / len(allm),
                  "margin_answer": sum(ansm) / len(ansm) if ansm else None}))
