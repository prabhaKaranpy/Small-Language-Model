# Small Language Model for TinyStories

A compact, decoder-only Transformer-based language model designed to generate coherent short stories using the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. 
Built with **PyTorch**, optimized for **GPU training**, and powered by **Byte-Pair Encoding (BPE)** tokenization.

---

## Overview

This project implements a Small Language Model (SLM) capable of generating short, coherent stories (~800–1000 words). It leverages:

- A decoder-only Transformer architecture (GPT-style).
- BPE tokenization via [`tiktoken`](https://github.com/openai/tiktoken).
- The `roneneldan/TinyStories` dataset from HuggingFace.

---

## Features

- ✅ **Short Story Generation** from minimal prompts.
- 🧠 **Decoder-Only Transformer** (GPT-based).
- 🔤 **Sub-word Tokenization** using Byte-Pair Encoding.
- 📈 **+20% Accuracy Boost** via extended training.
- ⚡ **GPU Acceleration** (Tested on RTX 3050 Laptop GPU).
- 🔧 **Full PyTorch Implementation**.
- 🎯 **Mixed Precision Training** (bfloat16/float16).

---

## 🛠Model & Training Configuration

### Dataset
- **Source**: `roneneldan/TinyStories` via HuggingFace Datasets.

### Tokenizer
- **Library**: `tiktoken`
- **Encoding**: `gpt2`

### Model Architecture (`GPTConfig`)

| Hyperparameter | Value     |
|----------------|-----------|
| `vocab_size`   | 50257     |
| `block_size`   | 128       |
| `n_layer`      | 6         |
| `n_head`       | 6         |
| `n_embd`       | 384       |
| `dropout`      | 0.1       |
| `bias`         | True      |

### 🎯 Training Parameters

| Parameter                     | Value                  |
|-------------------------------|------------------------|
| `learning_rate`               | 1e-4                   |
| `min_lr`                      | 5e-4                   |
| `max_iters`                   | 30,000                 |
| `warmup_steps`                | 1,000                  |
| `eval_iters`                  | 500                    |
| `batch_size`                  | 32                     |
| `gradient_accumulation_steps`| 32                     |
| `optimizer`                   | AdamW (β=(0.9, 0.95))  |
| `weight_decay`                | 0.1                    |
| `lr_scheduler`                | SequentialLR (Linear Warmup + Cosine Decay) |
| `precision`                   | Mixed (bfloat16/float16) |

---

## 📦 Installation

### ✅ Prerequisites

Make sure you have Python 3.x and install the following packages:

```bash
pip install torch datasets tiktoken numpy tqdm
