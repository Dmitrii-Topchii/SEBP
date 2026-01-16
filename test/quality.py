import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import bitsandbytes as bnb
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import lm_eval
from lm_eval.models.huggingface import HFLM
from sebp import apply_sebp_to_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print(f"Running on: {torch.cuda.get_device_name(0)}")
clear_gpu()

def evaluate_model(model, tokenizer, limit=1000):
    task_list = ['winogrande', 'arc_easy', 'arc_challenge', 'hellaswag']
    print(f"\nMeasuring Quality on {task_list} (Limit={limit})...")

    clear_gpu()
    model.eval()

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    results = lm_eval.simple_evaluate(
        model=lm_obj, tasks=task_list, limit=limit, num_fewshot=0
    )

    summary = {}
    for task, metrics in results['results'].items():
        val = metrics.get('acc_norm,none') or metrics.get('acc,none') or metrics.get('acc_norm') or metrics.get('acc') or 0.0
        summary[task] = float(f"{val:.4f}")

    model.train()
    print(f"Scores: {summary}")

    del lm_obj
    del results
    clear_gpu()

    return summary

def train_loop(model, tokenizer, steps=50, seq_len=1024):
    print(f"Training for {steps} steps (SeqLen={seq_len})...")

    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    dataset = load_dataset("text", data_files={"train": url}, split="train")
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)

    losses = []
    model.train()

    for i in range(steps):
        if i % 10 == 0: clear_gpu()

        text_batch = dataset[i:i+1]['text']
        text_batch = [t for t in text_batch if len(t) > 0]
        if not text_batch: continue

        inputs = tokenizer(
            text_batch, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True
        ).to(device)

        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")

    return losses

if __name__ == "__main__":
    model_id = "unsloth/Llama-3.2-3B-Instruct"
    LIMIT_VAL = 1000
    SEQ_LEN = 2048
    STEPS = 50

    print("\n" + "="*50)
    print("PHASE 1: BASELINE RUN")
    print("="*50)
    clear_gpu()
    model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model_base.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    losses_base = train_loop(model_base, tokenizer, steps=STEPS, seq_len=SEQ_LEN)

    del model_base
    clear_gpu()

    print("\n" + "="*50)
    print("PHASE 2: SEBP RUN")
    print("="*50)

    model_sebp = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model_sebp.gradient_checkpointing_enable()

    scores_pre = evaluate_model(model_sebp, tokenizer, limit=LIMIT_VAL)

    print(f"\nInjecting SEBP...")
    apply_sebp_to_model(model_sebp, sparse_rows=400)

    losses_sebp = train_loop(model_sebp, tokenizer, steps=STEPS, seq_len=SEQ_LEN)

    scores_post = evaluate_model(model_sebp, tokenizer, limit=LIMIT_VAL)

    print("\n" + "="*65)
    print("FINAL REPORT: SEBP VALIDATION")
    print("="*65)
    print(f"{'Dataset':<20} | {'Before FT':<12} | {'After FT':<12} | {'Delta':<10}")
    print("-" * 65)

    for task in scores_pre.keys():
        val_pre = scores_pre.get(task, 0.0)
        val_post = scores_post.get(task, 0.0)
        delta = val_post - val_pre
        symbol = "ok" if delta >= -0.01 else "not ok"
        print(f"{task:<20} | {val_pre:<12.4f} | {val_post:<12.4f} | {delta:+.4f} {symbol}")
    print("="*65)

    plt.figure(figsize=(10, 6))
    plt.plot(losses_base, label='Baseline', linestyle='--', color='blue', linewidth=2, alpha=0.7)
    plt.plot(losses_sebp, label='SEBP', color='green', linewidth=2)

    plt.title(f'Training Convergence: Baseline vs SEBP\n(Llama 3.2 3B | SeqLen={SEQ_LEN})')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_name = "SEBP_vs_Baseline_Quality.png"
    plt.savefig(plot_name)
    print(f"Plot saved to: {plot_name}")