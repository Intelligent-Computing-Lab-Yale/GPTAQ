
<h1 align="center">  
    <p> GPTQv2: Efficient Finetuning-Free Quantization with Asymmetric Calibration </p>  
</h1>  
  
<h1 align="center">   
    <img src="./img/readme_intro.png" width="1000">  
</h1>  
  
The official pytorch implementation of GPTQv2.   
  
Unlike the previous GPTQ method, which independently calibrates each layer, we always match the quantized layer’s output to the exact output in the full-precision model, resulting in a scheme that we call asymmetric calibration. Such a scheme can effectively reduce the quantization error accumulated in previous layers. We analyze this problem using optimal brain compression to derive a close-formed solution. The new solution explicitly minimizes the quantization error as well as the accumulated asymmetry error. Furthermore, we utilize various techniques to parallelize the solution calculation, including channel parallelization, neuron decomposition, and Cholesky reformulation for matrix fusion. As a result, GPTQv2 is easy to implement, simply using 20 more lines of code than GPTQ but improving its performance under low-bit quantization.  
  
  

## Update: GPTQv2 is integrated into GPTQModel

The GPTQv2 method is integrated into [GPTQModel](https://github.com/ModelCloud/GPTQModel/tree/main) library, with a simple argument to perform. 

You can install GPTQModel:

```shell
pip install -v gptqmodel --no-build-isolation
```

Quantize LLaMA3.1-8B-Instruct

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
quant_path = "Llama-3.1-8B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(256))["text"]

# Use GPTQv2 by passing v2=True
quant_config = QuantizeConfig(bits=4, group_size=128, v2=True)   

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

# test post-quant inference
model = GPTQModel.load(quant_path)
result = model.generate("Uncovering deep insights begins with")[0] # tokens
print(model.tokenizer.decode(result)) # string output
```


Performance comparison (GPTQv2 outperforms GPTQ on GSM8K using 1 fewer bit): 

| Method | Bits   | Arc_Challenge | GSM8K_Platinum_cot |
|--------|--------|---------------|--------------------|
| GPTQ   | W4g128 | 49.15         | 48.30              |
| GPTQv2 | W4g128 | 49.74         | **61.46**          |
| GPTQ   | W3g128 | 39.93         | 43.26              |
| GPTQv2 | W3g128 | 41.13         | **50.54**          |




## Code Structure  
  
We provide several directories to reproduce the paper results.   
  
1. [**fake_quant**](./fake_quant) for reproducing QuaRot+GPTQ/GPTQv2   
2. [**spinquant**](./spinquant) for reproducing SpinQuant+GPTQ/GPTQv2  
3. [**vit_quant**](./vit_quant) for reproducing vision transformer quantization results  

[//]: # (4. **GPTQModel**, a forked version of GPTQModel to support GPTQv2 to deploy weight-only quantization model  )
  
We recommend use separate envrionments for different experiments to ensure results are matched.


## Acknowledgement

Our code is built upon several repository:

[https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)

[https://github.com/spcl/QuaRot](https://github.com/spcl/QuaRot)

[https://github.com/facebookresearch/SpinQuant/tree/main](https://github.com/facebookresearch/SpinQuant/tree/main)


## Star Histroy
[![Star History Chart](https://api.star-history.com/svg?repos=Intelligent-Computing-Lab-Yale/GPTQv2&type=Date)](https://star-history.com/#Intelligent-Computing-Lab-Yale/GPTQv2)

## Contact

Yuhang Li (*yuhang.li@yale.edu*)

## Citations  
  
If you find GPTQv2 useful, please consider giving a star and citation:  
```bibtex  
@article{li2025gptqv2,
      title={GPTQv2: Efficient Finetuning-Free Quantization for Asymmetric Calibration}, 
      author={Yuhang Li and Ruokai Yin and Donghyun Lee and Shiting Xiao and Priyadarshini Panda},
      year={2025},
      journal={arXiv preprint arXiv:2504.02692},
}
```