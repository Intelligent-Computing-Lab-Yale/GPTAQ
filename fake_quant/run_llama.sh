#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

python main.py --model meta-llama/Meta-Llama-3-8B  \
 --w_bits 4 \
 --w_groupsize -1 \
 --w_clip \
 --a_bits 4 \
 --v_bits 16 \
 --k_bits 16 \
 --k_asym \
 --v_asym \
 --w_asym \
 --a_asym \
 --a_clip_ratio 0.9 \
 --k_clip_ratio 0.95 \
 --v_clip_ratio 0.95 \
 --use_v2 \
 --enable_aq_calibration \
 --rotate \