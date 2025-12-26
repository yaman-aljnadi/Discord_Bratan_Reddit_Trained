import os
import json
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
from tqdm import tqdm  # pip install tqdm

INPUT_DIR = "./processed_data_chat_best" 
MODEL_ID = "mistralai/Mistral-7B-v0.1"
MAX_WORKERS = 12 

tokenizer = None

def init_worker():
    """
    Initialize the tokenizer once per worker process to save overhead.
    """
    global tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer in worker: {e}")

def count_file_tokens(file_path):
    """
    Reads a single .jsonl file and counts tokens for instruction + response.
    """
    total_tokens = 0
    total_pairs = 0
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    instruction = entry.get('instruction', "")
                    response = entry.get('response', "")
                    
                    # Encode and count. 
                    # Note: We use add_special_tokens=False to count raw text content.
                    # If your training template adds <s> or [INST], those are overhead tokens.
                    t1 = len(tokenizer.encode(instruction, add_special_tokens=False))
                    t2 = len(tokenizer.encode(response, add_special_tokens=False))
                    
                    total_tokens += (t1 + t2)
                    total_pairs += 1
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        return (total_pairs, total_tokens)
    
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
        return (0, 0)

def get_jsonl_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')]

if __name__ == "__main__":
    # 1. Setup
    print(f"--- Dataset Token Calculator ({MODEL_ID}) ---")
    files = get_jsonl_files(INPUT_DIR)
    
    if not files:
        print(f"No .jsonl files found in {INPUT_DIR}")
        exit()
        
    print(f"Found {len(files)} files to process.")
    
    grand_total_pairs = 0
    grand_total_tokens = 0
    
    # 2. Parallel Processing
    # We use chunksize=1 to ensure smooth progress bar updates for large files
    with Pool(processes=MAX_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(count_file_tokens, files), 
            total=len(files), 
            unit="file"
        ))
    
    # 3. Aggregation
    for pairs, tokens in results:
        grand_total_pairs += pairs
        grand_total_tokens += tokens
        
    # 4. Final Report
    print("\n" + "="*40)
    print(f"FINAL STATS FOR: {INPUT_DIR}")
    print("="*40)
    print(f"Total Files Processed:  {len(files)}")
    print(f"Total Chat Pairs:       {grand_total_pairs:,}")
    print(f"Total Raw Tokens:       {grand_total_tokens:,}")
    
    # Estimate size in Billions
    print(f"Size in Billions (B):   {grand_total_tokens / 1e9:.4f} B")
    
    # Calculate Average tokens per context (helpful for setting seq_len)
    if grand_total_pairs > 0:
        avg_len = grand_total_tokens / grand_total_pairs
        print(f"Avg Tokens per Pair:    {avg_len:.2f}")
    print("="*40)