import os
import torch
import inspect
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
DATA_PATH = "/home/data/files/Yaman/Discord_Bratan_Reddit_Trained/Data_Cleaning_And_Processing/processed_data_chat_best/"
OUTPUT_DIR = "./mistral-reddit-15B-v1"
MODEL_ID = "mistralai/Mistral-7B-v0.1"

# --- V100 OPTIMIZED HYPERPARAMETERS ---
BATCH_SIZE = 2        
GRAD_ACCUMULATION = 8 
LEARNING_RATE = 1e-4    
NUM_EPOCHS = 1        
MAX_SEQ_LENGTH = 2048   

def formatting_func(example):
    instr = example.get('instruction', '')
    resp = example.get('response', '')
    return f"[INST] {instr} [/INST] {resp}"

def main():
    print(f"--- Starting Training Job on Tesla V100 (Single GPU) ---")
    
    # 1. Load Dataset
    dataset = load_dataset("json", data_dir=DATA_PATH, split="train[:10%]")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token  
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    # 3. Load Model (Optimized for V100 - FP16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
    )

    # Use GPU 0 directly since we are running with python (not torchrun)
    device_map = {"": 0}

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,
        attn_implementation="eager",
        dtype=torch.float16 
    )
    
    # --- MANUAL MODEL PREPARATION (The Fix) ---
    
    # 1. Enable gradients
    model.enable_input_require_grads()
    
    # 2. Define LoRA Config
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

    # 3. Manually apply LoRA (so we can sanitize afterwards)
    print("Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)

    # 4. --- SANITIZATION LOOP ---
    # This iterates over every parameter. If it finds BFloat16, it kills it.
    print("Sanitizing model dtypes for V100 compatibility...")
    count_converted = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            # print(f"Converting {name} from bf16 to fp16")
            param.data = param.data.to(torch.float16)
            count_converted += 1
    
    print(f"Sanitized {count_converted} parameters from BFloat16 to Float16.")

    # 5. Force Norms to Float32 (Standard stability fix)
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)
            
    # Verify Trainable Parameters
    model.print_trainable_parameters()

    # 6. SFT Config
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        optim="paged_adamw_32bit",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500, 
        learning_rate=LEARNING_RATE,
        bf16=False,   # Strictly False
        fp16=True,    # Strictly True
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        dataset_text_field="text",
        # Important: Since we wrapped manually, we tell SFTTrainer we already did it
    )

    args.max_seq_length = MAX_SEQ_LENGTH
    args.packing = False  

    # 7. Initialize Trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        # "peft_config": peft_config,  <-- REMOVED! We already applied it manually.
        "formatting_func": formatting_func,
        "args": args,
    }

    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {os.path.abspath(OUTPUT_DIR)}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()