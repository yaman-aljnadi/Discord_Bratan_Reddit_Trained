import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from trl import SFTTrainer, SFTConfig

# --- Configuration ---
# Point this to your output directory from the previous script
DATA_PATH = "/home/yaljnadi/Desktop/Discord_Bratan_Reddit_Trained/Data_Cleaning_And_Processing/processed_data_chat_best/" 
OUTPUT_DIR = "./mistral-reddit-15B-v1"
MODEL_ID = "mistralai/Mistral-7B-v0.1"

# Hyperparameters for 15B tokens
# We use a larger batch size via accumulation to stabilize the massive data intake
BATCH_SIZE = 16        # Adjust based on VRAM (try 8 or 16 per GPU)
GRAD_ACCUMULATION = 2  # effective batch = batch_size * num_gpus * grad_accum
LEARNING_RATE = 1e-4   # QLoRA standard (higher than full ft)
NUM_EPOCHS = 1         # With 15B tokens, 1 epoch is likely enough to shift the style totally
MAX_SEQ_LENGTH = 2048  # Mistral context window

def formatting_func(example):
    """
    Formats the input into the Mistral instruction format.
    Format: <s>[INST] {instruction} [/INST] {response}</s>
    """
    # Note: The tokenizer usually adds <s> automatically, so we handle the rest
    text = f"[INST] {example['instruction']} [/INST] {example['response']}"
    return [text]

def main():
    print(f"--- Starting Training Job ---")
    print(f"Loading data from: {DATA_PATH}")

    # 1. Load Dataset
    # using 'streaming=True' is safer for 15B tokens if you have limited System RAM (128GB)
    # However, for packing=True efficiency, we ideally want map-style.
    # The 'datasets' library uses memory mapping (Apache Arrow), so it won't load 
    # all 60GB+ of text into RAM at once, it reads from disk.
    dataset = load_dataset("json", data_dir=DATA_PATH, split="train")
    
    print(f"Dataset loaded. Rows: {len(dataset)}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token  # Mistral specific hack
    tokenizer.padding_side = "right" # Fix for fp16

    # 3. Load Model in 4-bit (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,  # Disable cache for training
        attn_implementation="flash_attention_2" # MANDATORY for speed on DGX
    )
    
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    # We target all linear layers to ensure the "personality" is deeply ingrained
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,   # Rank 64 is robust for large datasets
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        save_strategy="steps",
        save_steps=2000,
        learning_rate=LEARNING_RATE,
        bf16=True,             # Excellent for Blackwell
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        dataset_text_field=None, 
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        args=args,
        # MOVED BACK HERE:
        max_seq_length=MAX_SEQ_LENGTH, 
        packing=True                   
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    
    # Save tokenizer for inference convenience
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()