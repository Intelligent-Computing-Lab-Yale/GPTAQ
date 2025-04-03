# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

export HF_HUB_CACHE="/gpfs/gibbs/pi/panda/yl2447/llmc/cache"

find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)


#MASTER_ADDR=127.0.0.1
#MASTER_PORT=$UNUSED_PORT
#
#torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
# ptq.py \
#--input_model "meta-llama/Meta-Llama-3-8B" \
#--do_train False \
#--do_eval  True \
#--per_device_eval_batch_size 4 \
#--model_max_length 2048 \
#--fp16 False \
#--bf16 True \
#--save_safetensors False \
#--w_bits 4 \
#--a_bits 4 \
#--k_bits 16 \
#--v_bits 16 \
#--w_clip \
#--a_asym \
#--k_asym \
#--v_asym \
#--k_groupsize 128 \
#--v_groupsize 128 \
#--rotate \
#--optimized_rotation_path "ckpts/8B_W16A4KV16_lr_1.5_seed_0/R.bin" \
#--lm_eval \
#--lm_eval_batch_size 32 \
#--use_v2 \
#--enable_aq_calibration \

#meta-llama/Llama-2-70b-hf




torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model "meta-llama/Meta-Llama-3-8B" \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
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
