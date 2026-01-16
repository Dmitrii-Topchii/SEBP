import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 1
NUM_SAMPLES = 50 

def get_hf_token():
    return os.environ.get("HF_TOKEN")

def load_data_generator(task_name):
    if task_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return [x["text"] for x in ds if len(x["text"]) > 10]
    
    elif task_name == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="train")
        return [x["sentence"] for x in ds if len(x["sentence"]) > 10]

    elif task_name == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", split="train")
        return [x["sentence"] for x in ds]

    elif task_name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        return [f"Question: {x['question']} Answer: {x['answerKey']}" for x in ds]

    elif task_name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        return [f"Question: {x['question']} Answer: {x['answerKey']}" for x in ds]

    elif task_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="train")
        return [f"{x['ctx']} {x['endings'][0]}" for x in ds]
    
    return []

def row_sparsity_exact(g):
    if g is None: return float("nan")
    g2 = g.reshape(-1, g.shape[-1])
    zeros = (g2 == 0).all(dim=-1)
    return float(zeros.float().mean().item())

def attach_hooks(model, store):
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "layers" in name:
            m = re.search(r"\blayers\.(\d+)\b", name)
            if m:
                layer_idx = int(m.group(1))
                def hook(module, grad_in, grad_out, idx=layer_idx):
                    if grad_out and grad_out[0] is not None:
                        store[idx]["out"].append(row_sparsity_exact(grad_out[0].detach()))
                    if grad_in and grad_in[0] is not None:
                        store[idx]["inter"].append(row_sparsity_exact(grad_in[0].detach()))
                handles.append(mod.register_full_backward_hook(hook))
    return handles

def measure_dataset(model, tokenizer, task_name):
    texts = load_data_generator(task_name)
    texts = texts[:NUM_SAMPLES]
    
    store = {} 
    for i in range(len(model.model.layers)):
        store[i] = {"inter": [], "out": []}
        
    handles = attach_hooks(model, store)
    
    model.train()
    model.zero_grad()

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding=False).to(DEVICE)
        if inputs.input_ids.shape[1] < 2: continue
        
        outputs = model(**inputs, labels=inputs.input_ids)
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)

    for h in handles: h.remove()

    inter_vals = []
    out_vals = []
    for layer_idx in store:
        if store[layer_idx]["inter"]:
            inter_vals.append(np.mean(store[layer_idx]["inter"]))
        if store[layer_idx]["out"]:
            out_vals.append(np.mean(store[layer_idx]["out"]))
            
    return np.mean(inter_vals), np.mean(out_vals)

if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=get_hf_token())
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=get_hf_token(), torch_dtype=DTYPE, device_map="auto"
    )

    tasks = ["wikitext", "ptb", "winogrande", "arc_easy", "arc_challenge", "hellaswag"]
    results = []

    for task in tasks:
        print(f"Processing {task}...")
        try:
            inter, out = measure_dataset(model, tokenizer, task)
            results.append({
                "Dataset": task,
                "Intermediate Sparsity": inter,
                "Output Sparsity": out
            })
            print(f"  {task}: Inter={inter:.2%}, Out={out:.2%}")
        except Exception as e:
            print(f"  Error on {task}: {e}")

    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL REPORT (Strict Zeros)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/sparsity_multitask.csv", index=False)