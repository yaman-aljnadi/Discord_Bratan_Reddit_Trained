import os
import bz2
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = "Reddit_Data/My_250GB_Reddit_Data/2007"  # Your root folder
OUTPUT_DIR = "./processed_data" # Where to save the clean files
MIN_SCORE = 3       # Only keep comments with upvotes > 3
MIN_WORDS = 5       # Filter out "lol", "thanks", etc.
MAX_WORDS = 1000    # Filter out massive copy-pastes/spam
BATCH_SIZE = 10000  # How many lines to write at once

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """
    Basic text cleaning.
    """
    # Remove URL links (simple regex)
    text = re.sub(r'http\S+', '', text)
    # Remove multiple newlines/spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(file_path):
    """
    Reads a single .bz2 file, filters content, and writes to a .jsonl file.
    """
    file_name = os.path.basename(file_path)
    output_file = os.path.join(OUTPUT_DIR, file_name.replace('.bz2', '_clean.jsonl'))
    
    # Skip if already processed
    if os.path.exists(output_file):
        return f"Skipped {file_name}"

    valid_rows = []
    
    try:
        # bz2.open allows streaming the compressed file directly
        with bz2.open(file_path, "rt", encoding="utf-8") as source, \
             open(output_file, "w", encoding="utf-8") as target:
            
            for line in source:
                try:
                    data = json.loads(line)
                    
                    # --- FILTERING LOGIC ---
                    
                    # 1. Skip deleted/removed
                    body = data.get('body', '')
                    if body in ['[deleted]', '[removed]']:
                        continue
                        
                    # 2. Score Filter (Quality Control)
                    if data.get('score', 0) < MIN_SCORE:
                        continue
                        
                    # 3. Length Filter
                    word_count = len(body.split())
                    if word_count < MIN_WORDS or word_count > MAX_WORDS:
                        continue
                    
                    # 4. Content Cleaning
                    clean_body = clean_text(body)
                    
                    # Prepare the object for training
                    # We keep subreddit info as it might be useful for prompt steering later
                    training_entry = {
                        "text": clean_body,
                        "meta": {
                            "subreddit": data.get('subreddit'),
                            "score": data.get('score'),
                            "id": data.get('id')
                        }
                    }
                    
                    valid_rows.append(json.dumps(training_entry))
                    
                    # Write in batches to save RAM
                    if len(valid_rows) >= BATCH_SIZE:
                        target.write('\n'.join(valid_rows) + '\n')
                        valid_rows = []
                        
                except (json.JSONDecodeError, KeyError):
                    continue

            # Write remaining rows
            if valid_rows:
                target.write('\n'.join(valid_rows) + '\n')
                
        return f"Completed {file_name}"
        
    except Exception as e:
        return f"Error processing {file_name}: {str(e)}"

def get_all_files(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".bz2"):
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == '__main__':
    # 1. Gather all files
    files_to_process = get_all_files(SOURCE_DIR)
    print(f"Found {len(files_to_process)} files to process.")
    
    # 2. Utilize DGX Cores
    # Leave 2 cores free for system stability
    num_workers = max(1, cpu_count() - 2)
    print(f"Starting processing pool with {num_workers} workers...")
    
    # 3. process in parallel
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(process_file, files_to_process):
            print(result)