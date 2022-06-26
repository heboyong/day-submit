#!/usr/bin/env bash


GPUS=8
PORT=${PORT:-29506}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_submit.py --launcher pytorch ${@:4}  \
    --config work_dirs/week/week.py \
    --checkpoint work_dirs/week/latest.pth \
    --out submit_last/result.pkl


python test_submit_pkl.py

zip submit_last/predictions.zip submit_last/predictions.pkl