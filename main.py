import torch
import time
import sys
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from sebp import apply_sebp_to_model

def get_real_data(model_type, tokenizer, max_samples=100):

    print(f"Downloading dataset for {model_type}...")
    
    if model_type in ["llama", "qwen"]:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_datasets = tokenized_datasets.filter(lambda x: len(x['input_ids']) > 0)
        return tokenized_datasets.select(range(min(len(tokenized_datasets), max_samples)))

    elif model_type in ["bert", "roberta"]:
        dataset = load_dataset("glue", "mrpc", split="train")
        def tokenize_function(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], 
                           truncation=True, max_length=128, padding="max_length")
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets.select(range(min(len(tokenized_datasets), max_samples)))
    
    return None

def train_loop(model, dataset, model_type, steps=10):

    print(f"\nStarting Training ({steps} steps)...")
    
    model.train()
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)
    except ImportError:
        print("   (bitsandbytes not found, using standard AdamW)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch]).cuda()
        attention_mask = torch.tensor([item['attention_mask'] for item in batch]).cuda()
        labels = input_ids.clone() if model_type in ["llama", "qwen"] else None 
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    total_time = 0
    start_mem = torch.cuda.memory_allocated()
    
    for i, batch in enumerate(loader):
        if i >= steps: break
        
        optimizer.zero_grad()
        t0 = time.time()
        
        if model_type in ["llama", "qwen"]:
            outputs = model(input_ids=batch['input_ids'], 
                          attention_mask=batch['attention_mask'], 
                          labels=batch['labels'])
        else:
            outputs = model(input_ids=batch['input_ids'], 
                          attention_mask=batch['attention_mask'])

        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            loss = outputs.logits.sum() 
        
        loss.backward()
        optimizer.step()
        
        t1 = time.time()
        step_time = (t1 - t0) * 1000
        total_time += step_time
        
        print(f"   Step {i+1}/{steps} | Loss: {loss.item():.4f} | Time: {step_time:.2f}ms")

    end_mem = torch.cuda.memory_allocated()
    print(f"\n Validation Complete.")
    print(f"   Avg Step Time: {total_time/steps:.2f} ms")
    print(f"   Memory Delta:  {(end_mem - start_mem)/1024**2:.2f} MB")

if __name__ == "__main__":
    print("Hugging Face Authentication (Press Enter if already logged in)")
    token = input("Enter HF Token: ").strip()
    if token:
        login(token=token)

    print("\nSelect Model Architecture to Benchmark:")
    print("1. Llama 3.2 (CausalLM)")
    print("2. Qwen/Owen 2.5 (CausalLM)")
    print("3. BERT (MaskedLM)")
    print("4. RoBERTa (MaskedLM)")
    choice = input("Choice (1-4): ").strip()

    if choice == "1":
        model_id = "unsloth/Llama-3.2-3B-Instruct" 
        m_type = "llama"
    elif choice == "2":
        model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
        m_type = "qwen"
    elif choice == "3":
        model_id = "bert-base-uncased"
        m_type = "bert"
    elif choice == "4":
        model_id = "roberta-base"
        m_type = "roberta"
    else:
        print("Invalid choice")
        sys.exit()

    print(f"\nLoading Tokenizer & Data for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    real_dataset = get_real_data(m_type, tokenizer)
    
    print(f"--- Loading Model {model_id} ---")
    if m_type in ["llama", "qwen"]:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()

    print(f"\nInjecting SEBP")
    apply_sebp_to_model(model, sparse_rows=50)

    train_loop(model, real_dataset, m_type, steps=10)