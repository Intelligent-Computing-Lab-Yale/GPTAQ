
# SpinQuant

  

This repository contains the code of "[SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/pdf/2405.16406)"

 

## Run (Following original repo)

  

### 1. Requirements:

* python 3.9, pytorch >= 2.0

* install pytorch with cuda from https://pytorch.org/get-started/locally/, it is prerequisite for fast-hadamard-transform package.

* pip install -r requirement.txt

* git clone https://github.com/Dao-AILab/fast-hadamard-transform.git

* cd fast-hadamard-transform

* pip install .

### 2. Steps to run:
  
 **We directly use the pretrained rotation matrix of  spinquant to run our experiments**
 
You can download the optimized rotation matrices [here](https://drive.google.com/drive/folders/1R2zix4qeXBjcmgnJN1rny93cguJ4rEE8?usp=sharing).

**Note that to perform GPTQ/GPTQv2, we need to download the W16A4 rotation matrix.**

After obtaining the optimized_rotation, put the rotation matrix into optimized_rotation_path for evaluation.

* `bash scripts/2_eval_ptq.sh`

**For some reasons we don't know, we cannot reproduce exactly the same results with SpinQuant on LLaMA3, if you find out, please let us know**. For example, on LLaMA3-8B, we were unable to reproduce 7.10 perplexity. 


| Method                  | Bits | Perplexity |
|-------------------------|------|------------|
| SpinQuant+GPTQ (Paper)  | W4A4 | 7.1        |
| SpinQuant+GPTQ (Ours)   | W4A4 | 7.26       |
| SpinQuant+GPTQv2 (Ours) | W4A4 | 7.19       |

  

### 3. Export to ExecuTorch (The same with original code)

We also support exporting the quantized model to ExecuTorch, which allows us to utilize the quantization kernels and achieve real-time speedup. For more information on kernel implementation details, please see [ExecuTorch](https://pytorch.org/executorch/stable/index.html), and [ExecuTorch with SpinQuant](https://github.com/pytorch/executorch/tree/main/examples/models/llama#spinquant). We currently support 4-bit weight (set group-size to 256 for 8B model and to 32 for smaller model) and 8-bit dynamic activation quantization.

 
To obtain ExecuTorch-compatible quantized models, you can use the following scripts:

 
* `bash scripts/31_optimize_rotation_executorch.sh $model_name`

* `bash scripts/32_eval_ptq_executorch.sh $model_name`


### Arguments

  

- `--input_model`: The model name (or path to the weights)

- `--output_rotation_path`: The local path we want to store the oprimized rotation matrix

- `--per_device_train_batch_size`: The batch size for rotation optimization

- `--per_device_eval_batch_size`: The batch size for PPL evaluation

- `--a_bits`: The number of bits for activation quantization

- `--w_bits`: The number of bits for weight quantization

- `--v_bits`: The number of bits for value quantization

- `--k_bits`: The number of bits for key quantization

- `--w_clip`: Whether using the grid search to find best weight clipping range

- `--w_rtn`: Whether we want to use round-to-nearest quantization. If not having `--w_rtn`, we are using GPTQ quantization.

- `--w_groupsize`: The group size for group-wise weight quantization.

- `--rotate`: Whether we want to rotate the model

- `--optimized_rotation_path`: The checkpoint path of optimized rotation; Use random rotation if path is not given

  
