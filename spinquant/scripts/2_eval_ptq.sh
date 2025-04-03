# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.


torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model "meta-llama/Meta-Llama-3-8B" \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 16 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 4 \
--a_bits 4 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--optimized_rotation_path "ckpts/8B_W16A4KV16_lr_1.5_seed_0/R.bin" \
--use_v2 \ 
--enable_ap_calibration \
