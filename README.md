# Self-Instruct: Aligning LM with Self Generated Instructions

This repository contains code and data for the [Self-Instruct paper](https://arxiv.org/abs/2212.10560), a method for aligning pretrained language models with instructions.
This project extends the Self-Instruct framework by fine-tuning Qwen/Qwen2.5-0.5B-Instruct using QLoRA (Quantized Low-Rank Adaptation) for efficient instruction tuning.
The model is trained on the 82K Self-Instruct dataset (including 175 human-written seed tasks) and evaluated on the 252 expert-written, user-oriented Human Eval benchmark introduced in the original Self-Instruct paper.
Comprehensive evaluation and visualization scripts compare Qwen2.5-QLoRA and OpenAI Davinci models across BLEU, ROUGE-L, and BERTScore-F1 metrics, revealing the model’s instruction-following performance and generalization capability.
All fine-tuning, evaluation, and result-visualization steps are fully reproducible in this repository.

## Introduction

Self-Instruct is a framework that helps language models improve their ability to follow natural language instructions. It does this by using the model's own generations to create a large collection of instructional data. With Self-Instruct, it is possible to improve the instruction-following capabilities of language models without relying on extensive manual annotation.

### Background

In recent years, there has been a growing interest in building models that can follow natural language instructions to perform a wide range of tasks. These models, known as "instruction-tuned" language models, have demonstrated the ability to generalize to new tasks. However, their performance is heavily dependent on the quality and quantity of the human-written instruction data used to train them, which can be limited in diversity and creativity. To overcome these limitations, it is important to develop alternative approaches for supervising instruction-tuned models and improving their instruction-following capabilities.

### How Self-Instruct works?

The Self-Instruct process is an iterative bootstrapping algorithm that starts with a seed set of manually-written instructions and uses them to prompt the language model to generate new instructions and corresponding input-output instances. These generations are then filtered to remove low-quality or similar ones, and the resulting data is added back to the task pool. This process can be repeated multiple times, resulting in a large collection of instructional data that can be used to fine-tune the language model to follow instructions more effectively.

Here is an overview of Self-Instruct:

![The pipeline for generating instruction data from a language model itself.](docs/pipeline.JPG)

## Usage

\* **This work is still in progress. We may update the code and data as we make progress. Please be cautious about the version control.**

### Instruction-tuning using our Self-Instruct data

We release a dataset with 82K instance inputs and outputs. This instruction data can be used to conduct instruction-tuning for language models and make the language model follow instruction better. The entire model-generated data can be accessed in `data/gpt3-generations/batch_221203/all_instances_82K.jsonl`. This data (+ the 175 seed tasks) reformatted in clean GPT3-finetuning format (prompt + completion) is put in `data/finetuning/self_instruct_221203`. You can use the script in runner.ipynb.



### Evaluating instruction-following capabilities

We also release a new set of 252 expert-written tasks and their instructions motivated by user-oriented applications (rather than well-studied NLP tasks). This data is used in the human evaluation section of [the self-instruct paper](https://arxiv.org/abs/2212.10560). Please refer to [the human evaluation README](human_eval/README.md) for more details.

### Generating Self-Instruct data from scratch

To generate Self-Instruct data using your own seed tasks or other models, we open-source our scripts for the entire pipeline here. Our current code is only tested on the GPT3 model accessible via the [FIREWORKS API](accounts/fireworks/models/llama-v3p1-8b-instruct).


## Citation

This project extends the Self-Instruct framework proposed by
Wang et al., 2022, using Qwen/Qwen2.5-0.5B-Instruct as the base model and QLoRA for efficient 4-bit fine-tuning.
Our work reproduces the instruction-generation and evaluation pipeline, performs fine-tuning on the 82K instruction dataset, and evaluates results on the 252 user-oriented Human Eval benchmark, comparing Qwen2.5-QLoRA against OpenAI Davinci using BLEU, ROUGE-L, and BERTScore metrics.

```bibtex
@misc{selfinstruct,
  title={Self-Instruct: Aligning Language Model with Self Generated Instructions},
  author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2212.10560},
  year={2022}
}
```
