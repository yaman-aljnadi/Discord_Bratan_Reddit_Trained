import os
import bz2
import json
import re
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
from collections import defaultdict

# --- Configuration ---
MIN_SCORE = 3       
SOURCE_DIR = "Reddit_Data/My_250GB_Reddit_Data" 
OUTPUT_DIR = "./processed_data_best_only"
MIN_SCORE = 5        # STRICTER threshold (User requested 5)
MIN_WORDS = 3       
MAX_WORDS = 1000    
MAX_WORKERS = 12

# DGX Memory Management
# We need to limit workers because we are loading whole files into RAM now.
# With 128GB RAM, 10-12 workers is safe.
MAX_WORKERS = 12 

# Initialize Tokenizer (Global to avoid reloading)
# We use the Mistral tokenizer to get an exact count for your training planning.
MODEL_ID = "mistralai/Mistral-7B-v0.1" 
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
except:
    # Fallback if you haven't logged into HuggingFace or don't have net access
    print("Warning: Could not load Mistral Tokenizer. Using whitespace approximation.")
    tokenizer = None

def clean_text(text):
    """Basic cleaning."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_tokens(text):
    """Returns exact token count if tokenizer is loaded, else estimates."""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Rough estimate: 1 word ~= 1.3 tokens
        return int(len(text.split()) * 1.3)
def process_file_best_child(file_path):
    """
    1. Loads file.
    2. Groups children by Parent ID.
    3. Selects ONLY the highest scoring child for each parent.
    """
    file_name = os.path.basename(file_path)
    output_file = os.path.join(OUTPUT_DIR, file_name.replace('.bz2', '_best_chat.jsonl'))
    
    if os.path.exists(output_file):
        return (file_name, 0)

    # 1. Load Data & Group by Parent
    comment_map = {}          # Store all valid comments by their own ID
    children_by_parent = defaultdict(list) # Store list of children for every parent ID
    
    try:
        with bz2.open(file_path, "rt", encoding="utf-8") as source:
            for line in source:
                try:
                    data = json.loads(line)
                    
                    # Basic Cleanup Filters
                    if data.get('body') in ['[deleted]', '[removed]']: continue
                    
                    # We DON'T filter by score yet, because even a low score comment 
                    # might be a valid PARENT (Instruction).
                    
                    body = clean_text(data.get('body', ''))
                    if len(body.split()) < MIN_WORDS: continue
                    
                    c_id = data.get('id')
                    p_id = data.get('parent_id', '')
                    if p_id.startswith('t1_'): p_id = p_id[3:]
                    
                    score = data.get('score', 0)
                    
                    obj = {
                        "id": c_id,
                        "text": body,
                        "score": score,
                        "subreddit": data.get('subreddit'),
                        "parent_id": p_id
                    }
                    
                    # Store for lookup
                    comment_map[c_id] = obj
                    
                    # Grouping for "Sibling Competition"
                    children_by_parent[p_id].append(obj)
                    
                except (json.JSONDecodeError, KeyError):
                    continue

        # 2. The "Battle Royale" - Find Best Child
        chat_pairs = []
        
        # Iterate through every parent we found
        for p_id, children in children_by_parent.items():
            
            # Check if we actually have the Parent text (Instruction)
            if p_id in comment_map:
                parent_obj = comment_map[p_id]
                
                # SORT children by Score (Highest first)
                children.sort(key=lambda x: x['score'], reverse=True)
                
                winner = children[0]
                
                # 3. Apply the "Winner" Threshold
                if winner['score'] >= MIN_SCORE:
                    
                    entry = {
                        "instruction": parent_obj['text'],
                        "response": winner['text'],
                        "meta": {
                            "subreddit": winner['subreddit'],
                            "score": winner['score'],
                            "id": winner['id'],
                            "parent_id": p_id
                        }
                    }
                    chat_pairs.append(json.dumps(entry))

        # Write to disk
        if chat_pairs:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write('\n'.join(chat_pairs) + '\n')
        
        return (file_name, len(chat_pairs))

    except Exception as e:
        print(f"Error {file_name}: {e}")
        return (file_name, 0)

def get_all_files(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".bz2"):
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = get_all_files(SOURCE_DIR)
    print(f"Found {len(files)} files. Starting processing with {MAX_WORKERS} workers.")
    
    grand_total_tokens = 0
    grand_total_pairs = 0

    # Using imap to get results as they finish
    with Pool(MAX_WORKERS) as pool:
        for fname, pairs, tokens in pool.imap_unordered(process_file_best_child, files):
            grand_total_pairs += pairs
            grand_total_tokens += tokens
            if pairs > 0:
                print(f"Processed {fname}: {pairs} pairs | {tokens/1e6:.2f} M tokens")

    print("-" * 30)
    print("PROCESSING COMPLETE")
    print(f"Total Chat Pairs: {grand_total_pairs:,}")
    print(f"Total Tokens: {grand_total_tokens:,}")
    print(f"Estimated Training Data Size: {grand_total_tokens / 1e9:.4f} Billion Tokens")
    print("-" * 30)