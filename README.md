# SEBP: Acceleration of Backpropagation in Linear Layers of Transformer Models

## Overview

This repository contains the official implementation of **Sparsity-Exploiting Backward Pass (SEBP)**, a method for accelerating the fine-tuning of Transformer models. SEBP leverages the structured row sparsity found in output gradients- primarily caused by padding in short input sequences - to reduce the computational cost (FLOPs) of the backward pass in linear layers.

Unlike system-level optimizations that often increase memory consumption, SEBP provides significant speedups with negligible memory overhead by utilizing a custom Triton kernel to perform efficient dense-dense matrix multiplications on non-zero gradient rows.

## Key Contributions

* **Backward Pass Acceleration:** Achieves approximately **2.15x speedup** for BERT-base on GLUE tasks and **1.99x speedup** for LLaMA-3.2-3B on reasoning benchmarks.
* **Memory Efficiency:** Maintains a memory footprint comparable to standard PyTorch fine-tuning. For LLaMA-3B, SEBP requires only ~0.37 GB of additional memory, whereas alternative solutions like DeepSpeed can increase usage by over 5 GB.
* **Drop-In Integration:** The method is implemented as a custom `torch.autograd.Function` that overrides only the backward pass of linear layers, leaving the forward pass and model weights unchanged.
* **Architecture Support:** Validated on both encoder-only architectures (BERT, RoBERTa) and decoder-only architectures (Llama, Qwen).

## Methodology

In many NLP tasks, input sequences are significantly shorter than the model's maximum context length (e.g., 512 or 4096 tokens). Standard training pads these sequences to a fixed length, resulting in numerous zero-value tokens.

We observe that this padding induces structured row sparsity in the output gradients ($\frac{\partial L}{\partial Y}$) of linear layers. SEBP exploits this by:
1. **Sparsity Analysis:** Identifying rows in the gradient matrix that correspond to padding or contain negligible signal (noise below a calculated threshold $\epsilon$).
2. **Selective Computation:** Using a custom Triton kernel to select only the top-$k$ active rows.
3. **Dense Optimization:** Converting large sparse-dense multiplications into compact dense-dense operations, thereby skipping redundant computations for padding tokens.

## Repository Structure

```
SEBP/
├── sepb.py            # Core SEBP implementation (Triton kernel, autograd function, model patcher)
├── main.py            # Interactive benchmark script (Llama / Qwen / BERT / RoBERTa)
├── setup.py           # Package installation config
├── requirements.txt   # Python dependencies
├── test/
│   ├── quality.py     # Quality validation: lm-eval benchmarks before/after SEBP (Llama 3.2 3B)
│   └── sparsity.py    # Sparsity measurement: row-sparsity statistics across datasets
└── README.md
```

---

## Step-by-Step Launch Guide

### Step 1 — Prerequisites

Make sure you have:

| Requirement | Minimum |
|---|---|
| Python | 3.8+ |
| PyTorch | 2.1.0+ **with CUDA** |
| Triton | 2.1.0+ |
| GPU | NVIDIA with CUDA support (Ampere or newer recommended) |
| OS | Linux (recommended) or Windows with WSL |

> **Note:** Triton requires a CUDA-capable NVIDIA GPU. The library is not compatible with CPU-only or AMD environments.

### Step 2 — Clone the Repository

```bash
git clone https://github.com/Dmitrii-Topchii/SEBP.git
cd SEBP
```

### Step 3 — Create a Virtual Environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### Step 4 — Install Dependencies

**Option A** — via `requirements.txt` (recommended for running scripts):

```bash
pip install -r requirements.txt
```

**Option B** — via editable install (if you plan to import `sebp` as a package):

```bash
pip install -e .
```

Both options install all required libraries: PyTorch, Triton, Transformers, Datasets, etc.

### Step 5 — Authenticate with Hugging Face (if needed)

Some models (e.g., Llama) are gated and require a Hugging Face access token.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to **Settings → Access Tokens** and generate a token.
3. Accept the model license on the model page (e.g., [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)).

The script will prompt you for the token at launch, or you can log in beforehand:

```bash
huggingface-cli login
```

> **Tip:** BERT and RoBERTa do not require authentication — you can skip this step if using those models.

### Step 6 — Run the Interactive Benchmark

```bash
python main.py
```

You will be prompted to:

1. **Enter your HF token** (press Enter to skip if already logged in or using a public model).
2. **Select a model architecture:**

   ```
   1. Llama 3.2 (CausalLM)
   2. Qwen/Owen 2.5 (CausalLM)
   3. BERT (MaskedLM)
   4. RoBERTa (MaskedLM)
   ```

The script will then:
- Download the model and tokenizer from Hugging Face.
- Download a small evaluation dataset (WikiText-2 for causal LM, GLUE MRPC for masked LM).
- **Inject SEBP** into the model's linear layers.
- Run 10 training steps and report per-step loss, timing, and memory usage.

**Expected output** (example for BERT):

```
   SEBP Injection Complete.
   Replaced 74 layers.
   Configured Sparsity: Keep Top-50 rows.

Starting Training (10 steps)...
   Step 1/10 | Loss: 7.3241 | Time: 142.31ms
   ...
   Step 10/10 | Loss: 6.8102 | Time: 98.45ms

 Validation Complete.
   Avg Step Time: 105.67 ms
   Memory Delta:  12.34 MB
```

### Step 7 — Run Tests (optional)

#### 7a. Sparsity Measurement

Measures the actual row-sparsity of gradients across multiple benchmark datasets for Llama 3.2 3B:

```bash
python test/sparsity.py
```

This produces a CSV report in `results/sparsity_multitask.csv` with per-layer intermediate and output sparsity statistics.

#### 7b. Quality Validation

Evaluates model quality (via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)) before and after fine-tuning with SEBP injected, to confirm that sparsity-aware backpropagation does not degrade model performance:

```bash
pip install lm-eval
python test/quality.py
```

This runs WinoGrande, ARC-Easy, ARC-Challenge, and HellaSwag benchmarks and outputs a comparison table.

> **Note:** Quality validation requires significant GPU memory (~16 GB+) and time. An NVIDIA A100 or equivalent is recommended.

---

## Using SEBP in Your Own Code

You can integrate SEBP into any supported Transformer model in two lines:

```python
from sebp import apply_sebp_to_model

# Load your model normally
model = AutoModelForCausalLM.from_pretrained("your-model", torch_dtype=torch.float16).cuda()

# Inject SEBP — replaces linear layers with sparse-backward variants
apply_sebp_to_model(model, sparse_rows=50)

# Train as usual — the backward pass is now accelerated
```

**Parameters:**

| Parameter | Description | Default |
|---|---|---|
| `sparse_rows` | Number of top gradient rows to keep during backward pass. Higher values retain more gradient signal; lower values skip more computation. | `50` |
| `verbose` | Print injection summary. | `True` |

**Supported architectures:** Llama, Qwen, Mistral, BERT, RoBERTa (any model using standard HuggingFace naming conventions for linear layers).

---

## Contact

If any questions arise regarding the method, implementation, or reproduction of results, please reach out to:

**sayankotor@gmail.com**
