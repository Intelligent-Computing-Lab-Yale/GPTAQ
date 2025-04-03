

# Fake Quantization for Vision Transformers 
  
  This is code is developed based on our [`fake_quant`](../fake_quant)
  
## Installation

We can use the same envrionment with QuaRot. 

Additionally, install the `timm` package
  
## ImageNet Evaluations  
  
Currently, this code supports **EVA-02 and DeiT** transformers. The arguments are the same with `fake_quant` experiments.   

**Before you start, kindly modify the imagenet datapath in [data_utils.py](./data_utils.py) **
  
  
We provide a script `run.sh` to reproduce the results.

