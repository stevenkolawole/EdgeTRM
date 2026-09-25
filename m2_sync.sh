#!/bin/bash
# M2: pull index first, then cells (an index line is written only after its .npz), so every indexed cell is local.
cd "$(dirname "$0")" && rsync -a 'babel:m2_results/index_*.jsonl' results/m2/ && rsync -a babel:m2_results/cells/ results/m2/cells/
