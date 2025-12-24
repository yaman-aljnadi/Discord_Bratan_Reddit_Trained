import os
import bz2
import json
import re
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from time import time
from transformers import AutoTokenizer

# --- Configuration ---
SOURCE_DIR = "Reddit_Data/My_250GB_Reddit_Data" 
OUTPUT_DIR = "./processed_data_chat_best"

# Filtration Thresholds
MIN_SCORE = 3        # Strict quality control (Score must be >= 5)
MIN_WORDS = 5        # Skip "lol", "this", "yes"
MAX_WORDS = 1000     # Skip massive copy-pastes
MAX_WORKERS = 12     # Safe for 128GB RAM on DGX

# --- Tokenizer Initialization ---
# We load this globally so workers can access it via copy-on-write
MODEL_ID = "mistralai/Mistral-7B-v0.1" 
try:
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Mistral Tokenizer ({e}). Using whitespace approximation.")
    tokenizer = None

def clean_text(text):
    """
    Basic text cleaning to remove links and excessive whitespace.
    """
    if not text:
        return ""
    # Remove URL links
    text = re.sub(r'http\S+', '', text)
    # Remove multiple newlines/spaces and strip
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
    1. Reads .bz2 file.
    2. Groups comments by Parent ID.
    3. Selects HIGHEST SCORING child (>= MIN_SCORE).
    4. Creates Instruction/Response pair.
    5. Counts tokens for that pair.
    """
    file_name = os.path.basename(file_path)
    output_file = os.path.join(OUTPUT_DIR, file_name.replace('.bz2', '_best_chat.jsonl'))
    
    # Skip if already done
    if os.path.exists(output_file):
        return (file_name, 0, 0)

    # Data Structures
    comment_map = {}          # Maps ID -> Comment Data (used to retrieve "Instruction")
    children_by_parent = defaultdict(list) # Maps Parent_ID -> List of potential "Responses"
    
    try:
        # --- PHASE 1: LOAD & GROUP ---
        with bz2.open(file_path, "rt", encoding="utf-8") as source:
            for line in source:
                try:
                    data = json.loads(line)
                    
                    # 1. Basic Content Filter
                    body = data.get('body', '')
                    if body in ['[deleted]', '[removed]']: 
                        continue
                    
                    # 2. Length Filter
                    clean_body = clean_text(body)
                    word_count = len(clean_body.split())
                    
                    if word_count < MIN_WORDS or word_count > MAX_WORDS: 
                        continue
                    
                    # 3. Extract IDs
                    c_id = data.get('id')
                    p_id = data.get('parent_id', '')
                    
                    # Standardize Parent ID
                    if '_' in p_id:
                        p_id = p_id.split('_')[1]
                        
                    # 4. Store Data
                    score = data.get('score', 0)
                    
                    comment_obj = {
                        "id": c_id,
                        "text": clean_body,
                        "score": score,
                        "subreddit": data.get('subreddit'),
                        "parent_id": p_id
                    }
                    
                    # Save to map (Potential Instruction)
                    comment_map[c_id] = comment_obj
                    
                    # Save to grouping (Potential Response)
                    children_by_parent[p_id].append(comment_obj)
                    
                except (json.JSONDecodeError, KeyError):
                    continue

        # --- PHASE 2: SELECT BEST CHILD & COUNT TOKENS ---
        chat_pairs = []
        file_token_count = 0
        
        # Iterate through every parent ID we found children for
        for p_id, children in children_by_parent.items():
            
            # We can only create a pair if we have the Parent's text in this file
            if p_id in comment_map:
                parent_obj = comment_map[p_id]
                
                # Sort children by score (Highest first)
                children.sort(key=lambda x: x['score'], reverse=True)
                winner = children[0]
                
                # --- PHASE 3: FINAL QUALITY CHECK ---
                if winner['score'] >= MIN_SCORE:
                    
                    instruction_text = parent_obj['text']
                    response_text = winner['text']

                    # Calculate Tokens (Instruction + Response)
                    pair_tokens = count_tokens(instruction_text) + count_tokens(response_text)
                    file_token_count += pair_tokens
                    
                    # Create Entry
                    entry = {
                        "instruction": instruction_text,
                        "response": response_text,
                        "meta": {
                            "subreddit": winner['subreddit'],
                            "score": winner['score'],
                            "id": winner['id'],
                            "parent_id": p_id
                        }
                    }
                    chat_pairs.append(json.dumps(entry))

        # --- PHASE 4: WRITE TO DISK ---
        if chat_pairs:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write('\n'.join(chat_pairs) + '\n')
        
        return (file_name, len(chat_pairs), file_token_count)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return (file_name, 0, 0)

def get_all_files(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".bz2"):
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == '__main__':
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Gather files
    files_to_process = get_all_files(SOURCE_DIR)
    print(f"Found {len(files_to_process)} files to process.")
    print(f"Config: Min Score={MIN_SCORE}, Workers={MAX_WORKERS}")
    
    start_time = time()
    grand_total_pairs = 0
    grand_total_tokens = 0
    
    # Run Parallel Processing
    with Pool(MAX_WORKERS) as pool:
        for file_name, pair_count, token_count in pool.imap_unordered(process_file_best_child, files_to_process):
            grand_total_pairs += pair_count
            grand_total_tokens += token_count
            
            if pair_count > 0:
                print(f"Processed {file_name}: {pair_count} pairs | {token_count/1e6:.2f} M tokens")

    duration = time() - start_time
    print("-" * 30)
    print(f"PROCESSING COMPLETE in {duration:.2f} seconds")
    print(f"Total High-Quality Pairs: {grand_total_pairs:,}")
    print(f"Total Tokens: {grand_total_tokens:,}")
    print(f"Estimated Training Data Size: {grand_total_tokens / 1e9:.4f} Billion Tokens")
    print("-" * 30)