#!/usr/bin/env bash

stage=0
stop_stage=10000

data_dir=nictjle
cache_dir=cache/NICTJLE
glove_path=glove/glove.6B.300d.txt
save_root=output/NICTJLE
log_path=log/NICTJLE
sentaspara=para
model_type=HSG
problem_type=regression
mean_paragraphs=mean_residual
head=linear
wandb=

CUDA=3

. parse_options.sh
set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA \
        --data_dir $data_dir \
        --cache_dir $cache_dir \
        --embedding_path $glove_path \
        --model $model_type \
        --save_root $save_root \
        --log_root $log_path \
        --lr_descent \
        --grad_clip \
        -m 3 \
        --batch_size 30 \
        --sentaspara $sentaspara \
        --problem_type $problem_type \
        --mean_paragraphs $mean_paragraphs \
        --reweight \
        --head $head \
        --word_embedding \
        --wandb
fi
