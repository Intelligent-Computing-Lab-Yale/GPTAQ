

# Fake Quantization with QuaRot 
  
  This is code is developed based on [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://github.com/spcl/QuaRot)
  
## Installation

We recommend installing the python envrionments with original codebase's requirements:

```bash
conda create -n quarot python=3.9
conda activate quarot
pip install -r requirements.txt
```
Additionally, to apply Hadamard transformation, build [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) package from source.

  
## Language Generation and Zero-Shot Evaluations  
  
Currently, this code supports **LLaMa-2 and LLaMA-3** models (We did not test OPT and LLaMA-1).  
You can simply run the `main.py` to reproduce the results in the paper. The important arguments are:  
  
- `--model`: the model name (or path to the weights)  
- `--bsz`: the batch size for PPL evaluation  
- `--rotate`: whether we want to rotate the model  
- `--lm_eval`: whether we want to run LM-Eval for Zero-Shot tasks  
- `--tasks`: the tasks for LM-Eval  
- `--cal_dataset`: the calibration dataset for GPTQ quantization  
- `--a_bits`: the number of bits for activation quantization  
- `--w_bits`: the number of bits for weight quantization  
- `--v_bits`: the number of bits for value quantization  
- `--k_bits`: the number of bits for key quantization  
- `--w_clip`: Whether we want to clip the weights  
- `--a_clip_ratio`: The ratio of clipping for activation  
- `--k_clip_ratio`: The ratio of clipping for key  
- `--v_clip_ratio`: The ratio of clipping for value  
- `--w_asym`: Whether we want to use asymmetric quantization for weights  
- `--a_asym`: Whether we want to use asymmetric quantization for activation  
- `--v_asym`: Whether we want to use asymmetric quantization for value  
- `--k_asym`: Whether we want to use asymmetric quantization for key  
- `--a_groupsize`: The group size for activation quantization  
- `--w_groupsize`: The group size for weight quantization  
- `--v_groupsize`: The group size for value quantization  
- `--k_groupsize`: The group size for key quantization  
- `--use_v2`: Turn on GPTQv2 quantization (recommened)
- `--enable_aq_calibration`: Activation quantization during calibration (recommened)
  
  
We provide a script `run_llama.sh` to reproduce the results.

