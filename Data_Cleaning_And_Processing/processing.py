import os
import bz2
import json
import re
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

# --- Configuration ---
SOURCE_DIR = "Reddit_Data/My_250GB_Reddit_Data" 
OUTPUT_DIR = "./processed_data_chat"
MIN_SCORE = 3       
MIN_WORDS = 5       
MAX_WORDS = 1000    

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
    """Basic cleaning to remove links and whitespace."""
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

def process_file_pairing(file_path):
    """
    Reads a whole .bz2 file, builds a map, and links Parent -> Child.
    Returns: (File Name, Num Pairs Created, Total Tokens)
    """
    file_name = os.path.basename(file_path)
    output_file = os.path.join(OUTPUT_DIR, file_name.replace('.bz2', '_chat.jsonl'))
    
    if os.path.exists(output_file):
        return (file_name, 0, 0) # Skip if done

    # 1. Load Data into Memory (The "Map" Phase)
    comment_map = {} 
    
    try:
        with bz2.open(file_path, "rt", encoding="utf-8") as source:
            for line in source:
                try:
                    data = json.loads(line)
                    
                    # Basic Filtering
                    if data.get('body') in ['[deleted]', '[removed]']: continue
                    if data.get('score', 0) < MIN_SCORE: continue
                    
                    body = data.get('body', '')
                    word_count = len(body.split())
                    if word_count < MIN_WORDS or word_count > MAX_WORDS: continue

                    cleaned_body = clean_text(body)
                    
                    # Store in map: ID -> {Text, Parent_ID}
                    # We strip "t1_" from parent_id to match the 'id' format
                    c_id = data.get('id')
                    p_id = data.get('parent_id', '')
                    if p_id.startswith('t1_'):
                        p_id = p_id[3:] # Remove 't1_' prefix
                    
                    comment_map[c_id] = {
                        "text": cleaned_body,
                        "parent_id": p_id,
                        "subreddit": data.get('subreddit')
                    }
                    
                except (json.JSONDecodeError, KeyError):
                    continue

        # 2. Link Parents to Children (The "Reduce/Join" Phase)
        chat_pairs = []
        total_tokens = 0
        
        for c_id, child_data in comment_map.items():
            parent_id = child_data['parent_id']
            
            # Check if parent exists in our map
            if parent_id in comment_map:
                parent_data = comment_map[parent_id]
                
                # Format for Chat/Instruction Tuning
                # Instruction = Parent Comment
                # Response = Child Comment
                instruction = parent_data['text']
                response = child_data['text']
                
                # Calculate Tokens (Instruction + Response)
                pair_tokens = count_tokens(instruction) + count_tokens(response)
                total_tokens += pair_tokens

                entry = {
                    "instruction": instruction,
                    "response": response,
                    "meta": {
                        "subreddit": child_data['subreddit'],
                        "id": c_id,
                        "parent_id": parent_id
                    }
                }
                chat_pairs.append(json.dumps(entry))

        # 3. Write to Disk
        if chat_pairs:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write('\n'.join(chat_pairs) + '\n')
        
        return (file_name, len(chat_pairs), total_tokens)

    except Exception as e:
        print(f"Error in {file_name}: {e}")
        return (file_name, 0, 0)

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
        for fname, pairs, tokens in pool.imap_unordered(process_file_pairing, files):
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