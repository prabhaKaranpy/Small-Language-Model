Small Language Model for TinyStories
This project implements a Small Language Model (SLM) based on a decoder-only Transformer architecture to generate coherent short stories. The model is trained on the HuggingFace TinyStories dataset and utilizes Byte-Pair Encoding (BPE) for sub-word tokenization to handle text efficiently.

Features
Story Generation: Generates short stories ranging from 800-1000 words based on initial prompts.

Transformer Architecture: Built using a decoder-only Transformer model, a state-of-the-art architecture for sequence generation tasks.

Byte-Pair Encoding (BPE): Employs BPE for robust sub-word tokenization, managing vocabulary effectively and handling out-of-vocabulary words.

Performance Enhancement: Achieved a 20% increase in model accuracy through extended training over additional epochs.

PyTorch Implementation: Developed entirely in PyTorch, leveraging its powerful features for deep learning.

GPU Accelerated Training: Optimized for training on CUDA-enabled GPUs, specifically tested with an NVIDIA GeForce RTX 3050 A Laptop GPU.

Technical Details
The model's configuration and training parameters are as follows:

Dataset: roneneldan/TinyStories from HuggingFace

Tokenizer: tiktoken with gpt2 encoding

Model Configuration (GPTConfig):

vocab_size: 50257

block_size: 128

n_layer: 6

n_head: 6

n_embd: 384

dropout: 0.1

bias: True

Training Parameters:

learning_rate: 1e-4

max_iters: 30000

warmup_steps: 1000

min_lr: 5e-4

eval_iters: 500

batch_size: 32

gradient_accumulation_steps: 32

Optimizer: AdamW with betas=(0.9, 0.95) and weight_decay=0.1

Learning Rate Scheduler: SequentialLR combining Linear Warmup and Cosine Annealing LR Decay.

Mixed Precision Training: Uses bfloat16 or float16 for improved performance on compatible GPUs.

Getting Started
Prerequisites
Python 3.x

PyTorch

HuggingFace datasets library

tiktoken library

numpy

tqdm

You can install the required Python packages using pip:

pip install torch datasets tiktoken numpy tqdm

Data Preparation
The project automatically downloads and tokenizes the TinyStories dataset. It creates train.bin and validation.bin files in the working directory using Byte-Pair Encoding.

Training
To train the model, run the Small_Language_Model_30k_epochs.ipynb notebook. The notebook contains cells for setting up the environment, defining the model, configuring training parameters, and executing the training loop.

The training process includes:

GPU utilization check and setup

Dataset loading and tokenization

Model definition (GPT class and GPTConfig)

Training loop with loss estimation, learning rate scheduling, and mixed precision training

Story Generation
After training, the model can generate new stories using initial words. An example of story generation is provided in the notebook.

import tiktoken
import torch

# Assuming 'model' is your trained GPT model and 'enc' is your tiktoken encoder
enc = tiktoken.get_encoding("gpt2")
sentence = "Harry Potter and the Philosopher's Stone"
context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0))
y = model.generate(context, 300) # Generate 300 new tokens
print(enc.decode(y.squeeze().tolist()))
