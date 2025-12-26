import os
import torch
import inspect
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
DATA_PATH = "/home/yaljnadi/Desktop/Discord_Bratan_Reddit_Trained/Data_Cleaning_And_Processing/processed_data_chat_best/" 
OUTPUT_DIR = "./mistral-reddit-15B-v1"
MODEL_ID = "mistralai/Mistral-7B-v0.1"

# Hyperparameters
BATCH_SIZE = 16        
GRAD_ACCUMULATION = 2  
LEARNING_RATE = 1e-4   
NUM_EPOCHS = 1        
MAX_SEQ_LENGTH = 2048  

def formatting_func(example):
    """
    Formats the input into the Mistral instruction format.
    Format: <s>[INST] {instruction} [/INST] {response}</s>
    """
    # Safety check for empty fields to prevent training crashes
    instr = example.get('instruction', '')
    resp = example.get('response', '')
    text = f"[INST] {instr} [/INST] {resp}"
    return [text]

def get_compatible_args(cls, **kwargs):
    """
    Dynamic argument filtering:
    Inspects the class __init__ method and only passes arguments 
    that the class actually accepts.
    """
    sig = inspect.signature(cls.__init__)
    valid_args = {}
    for k, v in kwargs.items():
        if k in sig.parameters or 'kwargs' in sig.parameters:
            valid_args[k] = v
    return valid_args

def main():
    print(f"--- Starting Training Job on Blackwell GB10 ---")
    
    # 1. Load Dataset
    print(f"Loading data from: {DATA_PATH}")
    dataset = load_dataset("json", data_dir=DATA_PATH, split="train")
    print(f"Dataset loaded. Rows: {len(dataset)}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token  
    tokenizer.padding_side = "right"

    # 3. Load Model (Optimized for GB10)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model with SDPA (Blackwell Native Attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        attn_implementation="sdpa"  # STABLE for GB10 (since FlashAttn failed)
    )
    
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 5. Dynamic Configuration Construction
    # We define ALL intended arguments here
    all_config_args = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUMULATION,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_32bit",
        "logging_steps": 100,
        "save_strategy": "steps",
        "save_steps": 2000,
        "learning_rate": LEARNING_RATE,
        "bf16": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "report_to": "tensorboard",
        "ddp_find_unused_parameters": False,
        "dataset_text_field": "text",
        # These are the trouble makers:
        "max_seq_length": MAX_SEQ_LENGTH,
        "packing": True
    }

    # Filter args that SFTConfig ACTUALLY accepts
    safe_config_args = get_compatible_args(SFTConfig, **all_config_args)
    args = SFTConfig(**safe_config_args)

    # 6. Initialize Trainer (Dynamic)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "formatting_func": formatting_func,
        "args": args,
    }

    # Add max_seq_length/packing to Trainer if Config didn't take them
    if "max_seq_length" not in safe_config_args:
        print("DEBUG: Passing max_seq_length to Trainer directly.")
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    
    if "packing" not in safe_config_args:
        print("DEBUG: Passing packing to Trainer directly.")
        trainer_kwargs["packing"] = True

    # Handle the 'tokenizer' vs 'processing_class' rename
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        print("DEBUG: Using 'processing_class' argument.")
        trainer_kwargs["processing_class"] = tokenizer
    else:
        print("DEBUG: Using 'tokenizer' argument.")
        trainer_kwargs["tokenizer"] = tokenizer

    # Initialize
    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()