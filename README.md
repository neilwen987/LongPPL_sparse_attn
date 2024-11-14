# LongPPL

This repository is the official implementation for [What is Wrong with Perplexity for Long-context Language Modeling?](https://arxiv.org/abs/2410.23771)

## Introduction

Handling long-context inputs is crucial for large language models (LLMs). While recent approaches have extended the context windows of LLMs and employed perplexity (PPL) as a standard evaluation metric, PPL has proven unreliable for assessing long-context capabilities. We find that PPL overlooks key tokens, which are essential for long-context understanding, by averaging across all tokens and thereby obscuring the true performance of models in long-context scenarios. To address this, we propose `LongPPL`, a novel metric that focuses on key tokens by employing a long-short context contrastive method to identify them. Additionally, we introduce `LongCE` (Long-context Cross-Entropy) loss, a re-weighting strategy for fine-tuning that prioritizes key tokens.

<div align="center">    
    <img src="longppl.png" width = "600" height = "400" alt="LongPPL" align=center />
</div>

Our experiments demonstrate that LongPPL strongly correlates with performance on various long-context benchmarks (e.g., Pearson correlation of -0.96), significantly outperforming traditional PPL in predictive accuracy. Besides, experimental results also show that LongCE attains consistent improvements in a plug-and-play solution.

## Requirements
Python 3.10 + Pytorch 2.3 + Transformers 4.45

```
pip install -r requirements.txt
```

## LongPPL
The code support calculating LongPPL on customized LLMs and datasets. Please run:
```
pip install longppl
```
or 
```
git clone https://github.com/PKU-ML/LongPPL.git
cd LongPPL
pip install -e .
```

and use the following code to calculate LongPPL:

```
from longppl import compute_longppl

output = compute_longppl(text, model, evaluator_model, tokenizer, evaluator_tokenizer)
print(output['longppl'])
```

## Reproduce the paper
### LongPPL
To reproduce the LongPPL experiments in our paper, please run:
```
cd perplexity
sh run_ppl.sh
```
The evaluation data can be downloaded from [GovReport (tokenized)](https://huggingface.co/datasets/emozilla/govreport-test-tokenized). Here are our main results.

|Models|LongPPL(Qwen-72B-Instruct)|LongPPL(Mistral Large 2)|LongPPL(Llama-3.1-8B)|PPL|
|:---:|:---:|:---:|:---:|:---:|
|[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)|2.08|2.50|1.74|3.67|
|[FILM-7B](https://huggingface.co/In2Training/FILM-7B)|2.49|3.17|2.03|4.47|
|[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|2.68|3.49|2.19|4.25|
|[Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B)|2.97|2.93|2.33|5.23|
|[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)|2.99|2.73|2.29|4.97|
|[Phi-3-small](https://huggingface.co/microsoft/Phi-3-small-128k-instruct)|2.98|2.86|2.41|5.42|
|[CLEX-7B](https://huggingface.co/DAMO-NLP-SG/CLEX-LLaMA-2-7B-64K)|3.70|4.60|2.92|4.13|
|[Yi-6B](https://huggingface.co/01-ai/Yi-6B-200K)|3.62|3.92|2.86|5.11|
|[Yarn-7B](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)|3.67|4.88|3.10|4.17|

- While perplexity shows almost no correlation to their long-context performance measured by the benchmarks (please refer to our paper), LongPPL demonstrates a strong correlation.

### LongCE
To conduct long-context finetuning with LongCE, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training. 
```
cd finetune
sh train.sh
```
The training data can be downloaded from [PG19](https://huggingface.co/datasets/emozilla/pg19) and [Pile-arxiv](https://huggingface.co/datasets/suolyer/pile_arxiv).
To run models with eabf, please downgrade the version of `transformers` to `4.37.0`.

## Evaluation on Long-context Benchmark
In the paper, we evaluate models on [LongBench](https://github.com/THUDM/LongBench), [LongEval](https://github.com/DachengLi1/LongChat) and [RULER](https://github.com/nvtransfer/RULER). Please refer to the respective code repositories.

## Citation
If you use our code, please cite
```
@article{fang2024wrong,
      title={What is Wrong with Perplexity for Long-context Language Modeling?}, 
      author={Lizhe Fang and Yifei Wang and Zhaoyang Liu and Chenheng Zhang and Stefanie Jegelka and Jinyang Gao and Bolin Ding and Yisen Wang},
      year={2024},
      journal={arXiv preprint arXiv:2410.23771}
}
```