# SEBP: Acceleration of Backpropagation in Linear Layers of Transformer Models

## Overview

This repository contains the official implementation of **Sparsity-Exploiting Backward Pass (SEBP)**, a method for accelerating the fine-tuning of Transformer models. SEBP leverages the structured row sparsity found in output gradients—primarily caused by padding in short input sequences—to reduce the computational cost (FLOPs) of the backward pass in linear layers.

Unlike system-level optimizations that often increase memory consumption, SEBP provides significant speedups with negligible memory overhead by utilizing a custom Triton kernel to perform efficient dense-dense matrix multiplications on non-zero gradient rows.

## Key Contributions

* **Backward Pass Acceleration:** Achieves approximately **2.15x speedup** for BERT-base on GLUE tasks and **1.99x speedup** for LLaMA-3.2-3B on reasoning benchmarks.
* **Memory Efficiency:** Maintains a memory footprint comparable to standard PyTorch fine-tuning. For LLaMA-3B, SEBP requires only ~0.37 GB of additional memory, whereas alternative solutions like DeepSpeed can increase usage by over 5 GB.
* **Drop-In Integration:** The method is implemented as a custom `torch.autograd.Function` that overrides only the backward pass of linear layers, leaving the forward pass and model weights unchanged.
* **Architecture Support:** Validated on both encoder-only architectures (BERT, RoBERTa) and decoder-only architectures (Llama, Qwen).

## Methodology

In many Natural Language Processing (NLP) tasks, input sequences are significantly shorter than the model's maximum context length (e.g., 512 or 4096 tokens). Standard training pads these sequences to a fixed length, resulting in numerous zero-value tokens.

We observe that this padding induces structured row sparsity in the output gradients ($\frac{\partial L}{\partial Y}$) of linear layers. SEBP exploits this by:
1.  **Sparsity Analysis:** Identifying rows in the gradient matrix that correspond to padding or contain negligible signal (noise below a calculated threshold $\epsilon$).
2.  **Selective Computation:** Using a custom Triton kernel to select only the top-$k$ active rows.
3.  **Dense Optimization:** Converting large sparse-dense multiplications into compact dense-dense operations, thereby skipping redundant computations for padding tokens.

## Installation

### Prerequisites
* Python 3.8 or higher
* PyTorch >= 2.1.0 with CUDA support
* Triton >= 2.1.0
* NVIDIA GPU (Ampere architecture or newer recommended for optimal Triton performance)

### Setup
Clone the repository and install the package in editable mode:

```bash
git clone [https://github.com/Dmitrii-Topchii/SEBP.git)
cd sebp-acceleration
pip install -e .