# Enhancing Text Classification with Synthetic Data

This repository contains the code and resources related to my Bachelor Thesis in Artificial Intelligence:  
**"Enhancing Text Classification with Synthetic Data"**,  
by **Alessandro Ghiotto**, AY 2024/2025.

## 📚 Overview

Large Language Models (LLMs) such as GPT have shown exceptional performance across a wide range of NLP tasks. However, their high inference cost makes them less suitable for deployment in resource-constrained environments. This project explores a hybrid pipeline: using LLMs to **generate synthetic training data**, which is then used to fine-tune **efficient encoder-only classifiers** like RoBERTa.

Our results show that synthetic data can effectively improve classification performance, especially in low-resource settings, while maintaining a more efficient inference pipeline than relying on LLMs directly at runtime.

## 🔍 Research Questions

- Can synthetic labeled data from a decoder-only LLM train an encoder-only classifier effectively?
- How does training on synthetic data compare to real data?
- Does synthetic augmentation improve results over using real data alone?
- Is the proposed pipeline more computationally efficient than using LLMs for inference?

## 🗂️ Structure

```bash
bai-thesis-nlp/
├── src/
│   ├── _misc/          # miscellaneous notebooks and experimental files
│   ├── _utils/         # utility functions and helper code for data processing
│   ├── agnews/         # AG News dataset experiments and models
│   └── [other datasets]/  # individual dataset experiment folders
├── real_data/          # original real datasets
│   ├── train/          # training splits for all datasets
│   └── test/           # test splits for all datasets
├── synthetic_data/     # LLM-generated synthetic data
│   ├── logs/           # generation logs and metadata
│   └── datasets/       # processed synthetic datasets
├── papers/             # relevant research papers and references
├── images/             # plots, figures, and visualizations
└── _BAI__Thesis_Ghiotto_Alessandro/  # LaTeX thesis document
```

## 🧪 Experiments

We explored various synthetic data generation strategies:

1. **Baseline Prompting**
2. **Topic-Targeted Prompting**
3. **Prompting with Unsupervised Examples**
4. **Zero-Shot Labeling on Real Data**

Each dataset was used to fine-tune a RoBERTa classifier and compared against zero-/few-shot prompting and Adaptive In-Context Learning (AICL).

All experiments were conducted using small-scale open-source LLMs:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `meta-llama/Llama-2-7b-chat-hf`

## 📊 Key Findings

- **RoBERTa models fine-tuned on synthetic data** often performed competitively with those trained on real data.
- **Zero-shot labeling + fine-tuning** emerged as a promising approach in extremely low-resource settings.
- **LLM inference** (prompting) is viable but more computationally expensive than fine-tuned small models.

## ⚙️ Environment

To reproduce the results, create the conda environment:

```bash
conda env create -f environment.yaml
conda activate nlp-env
```

## 🤝 Acknowledgments

Supervised by:

- Alessandro Raganato – University of Milano-Bicocca
- Marco Braga – University of Milano-Bicocca

Thanks to the BAI program and the NLP research community for open-source resources and tools.
