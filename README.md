# UniBias

**UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation**

This repository contains the implementation for the NeurIPS 2024 paper that addresses bias in Large Language Models (LLMs) by identifying and mitigating biased components within the model's internal architecture.

## Overview

UniBias is a method for detecting and reducing bias in LLMs by analyzing and manipulating two key internal components:

1. **Attention Heads**: Identifies attention heads that exhibit biased behavior towards certain labels
2. **FFN (Feed-Forward Network) Neurons**: Identifies FFN neurons that contribute to label bias

The method uses a three-criterion approach to identify biased components:
- **Bias Criterion**: Components showing disproportionate preference for certain labels
- **Relatedness Criterion**: Components that are meaningfully related to the task
- **Low-Variance Criterion**: Components with consistent behavior across samples

## Key Features

- **Automatic Bias Detection**: Systematically identifies biased attention heads and FFN neurons through grid search
- **Internal Model Manipulation**: Directly modifies model behavior by masking or adjusting identified biased components
- **Multi-Task Support**: Works across various NLP tasks including sentiment analysis, natural language inference, question answering, and more
- **Calibration Evaluation**: Includes evaluation against baseline calibration methods (CC, DC, PC)

## Supported Datasets

- **Sentiment Analysis**: SST-2, SST-5, CR, MR
- **Natural Language Inference**: MNLI, RTE
- **Question Answering**: COPA, ARC, MMLU
- **Text Classification**: AG News, TREC, WiC

## Architecture

- `main.py`: Main entry point orchestrating the bias identification and mitigation pipeline
- `attention_manipulate.py`: Implements attention head bias identification and elimination
- `FFN_manipulate.py`: Implements FFN neuron bias identification and elimination
- `evaluation.py`: Handles model evaluation and calibration method comparisons
- `utils.py`: Dataset preparation, prompt generation, and utility functions

## How It Works

1. **Dataset Preparation**: Loads and prepares datasets with few-shot demonstrations
2. **Bias Identification**: 
   - Analyzes FFN neurons and attention heads using validation data
   - Applies grid search to find optimal thresholds for bias detection
3. **Bias Mitigation**: 
   - Masks or adjusts identified biased components
   - Optimizes debiasing parameters (alpha values)
4. **Evaluation**: 
   - Tests model performance on test sets
   - Compares against calibration baselines

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhou2024unibias,
  title={UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation},
  author={Zhou, Hanzhang and Feng, Zijian and Zhu, Zixiao and Qian, Junlang and Mao, Kezhi},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Paper

[NeurIPS 2024 Paper](https://arxiv.org/abs/2405.20612)

