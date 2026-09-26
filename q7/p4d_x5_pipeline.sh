#!/bin/bash
# X5 width-law model on p4d (one platform per model): train (fixed trainer, keep latest ckpt) -> S + blind prediction
# (sens_fit + x2_predict, log ~/breadth_results/x2/sens/predictions.jsonl) -> S_mix -> width sweep (q7_eval --full).
#   usage: p4d_x5_pipeline.sh RUN SEED [train args...]   (autofill sets CUDA_VISIBLE_DEVICES)
RUN=$1; SEED=$2; shift 2
source $HOME/q7/env_p4d.sh; export HF_HUB_OFFLINE=1 Q7_RUN=$RUN
C=$HOME/q7/ckpt/$RUN; O=$HOME/breadth_results/x2/sens; mkdir -p $O
bash $HOME/q7/q7_train2.sh $RUN $SEED "$@" +q7_inline_eval=False +q7_keep_last_ckpt=True || exit 1
CK=$(ls $C/step_* 2>/dev/null | grep -E "step_[0-9]+$" | sort -t_ -k2 -n | tail -1)
[ -f $C/DONE ] && [ -n "$CK" ] || { echo "no final checkpoint"; exit 1; }
cd $HOME/edgetrm_src
[ -f $O/sens_var_$RUN.json ] || { $Q7_PY sens_fit.py --family trm --task sudoku_mlp --ckpt $CK --variant $RUN --n 256 \
    --specs noise0.02:3,noise0.05:3,noise0.1:3,w8c,w4c,w4t,w3g32 --out $O/sens_var_$RUN.json && $Q7_PY x2_predict.py $O/sens_var_$RUN.json; } || exit 1
[ -f $O/smix_$RUN.json ] || $Q7_PY sens_fit.py --family trm --task sudoku_mlp --ckpt $CK --variant "$RUN@smix" --n 256 \
    --specs noise0.05@mixer:3,noise0.05@mlp:3 --out $O/smix_$RUN.json
$Q7_PY q7_eval.py --last --full --rundir $C
echo "$(date -u) [x5] $RUN pipeline done"
