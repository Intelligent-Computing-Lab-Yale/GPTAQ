#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id


python main.py --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k  \
 --w_bits 4 \
 --w_groupsize -1 \
 --w_clip \
 --a_bits 4 \
 --nsamples 128 \
 --a_asym \
 --w_asym \
 --percdamp 0.1 \
 --act_order \
 --bsz 256 \
 --asym_calibrate \
 --enable_aq_calibration \
