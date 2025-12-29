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
# BATCH_SIZE 8 is too high for V100 16GB with 4-bit Mistral + 2048 ctx.
# We lower batch size to 2 and increase accumulation to 8 to maintain effective batch size.
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
    print(f"--- Starting Training Job on Tesla V100 (Dual GPU DDP) ---")
    
    # 1. Load Dataset
    dataset = load_dataset("json", data_dir=DATA_PATH, split="train[:10%]")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token  
    tokenizer.padding_side = "right"

    # 3. Load Model (Optimized for V100 - FP16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
    )

    # DYNAMIC DEVICE MAPPING FOR MULTI-GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    print(f"Loading model on GPU {local_rank}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,
        attn_implementation="eager" # Critical for V100
    )
    
    # --- CRITICAL STABILITY FIX FOR V100 ---
    # Instead of prepare_model_for_kbit_training (which can be buggy with DDP on Volta),
    # we manually set up the model for training stability.
    
    # 1. Enable gradients on input embeddings (required for LoRA)
    model.enable_input_require_grads()
    
    # 2. Force LayerNorm to Float32 (Prevents Segfaults/NaNs on V100)
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

    # 3. LoRA Configuration
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

    # 4. SFT Config
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        
        # Gradient Checkpointing (Enabled via config only, to avoid conflicts)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 

        optim="paged_adamw_32bit",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500, 
        learning_rate=LEARNING_RATE,
        bf16=False,   
        fp16=True,    
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        ddp_find_unused_parameters=False, 
        dataset_text_field="text",
    )

    args.max_seq_length = MAX_SEQ_LENGTH
    args.packing = False  

    # 5. Initialize Trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
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

    if local_rank == 0:
        print(f"Saving final model to {os.path.abspath(OUTPUT_DIR)}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Done!")

if __name__ == "__main__":
    main()