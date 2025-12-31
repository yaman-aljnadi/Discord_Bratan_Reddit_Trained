import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

class MistralEngine:
    def __init__(self):
        # 1. PATHS
        # The base model you used for training
        self.base_model_id = "mistralai/Mistral-7B-v0.1"
        
        # Path to your specific checkpoint (Update the path if you move the files!)
        self.adapter_path = "/home/yaljnadi/Desktop/Discord_Bratan_Reddit_Trained/Training_Codes/mistral-reddit-15B-v1/checkpoint-500"

        print(f"Loading Base Model: {self.base_model_id}...")

        # 2. LOAD TOKENIZER
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True
        )
        # Fix padding token matching your training script
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"

        # 3. LOAD BASE MODEL (Quantized to 4-bit to fit in memory, same as training)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 4. ATTACH YOUR ADAPTER (The "Brain Implant")
        print(f"Loading Adapter from: {self.adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        print("Successfully loaded Custom Fine-Tuned Model!")

    def generate_response(self, user_input):
        # Your training code used this format: [INST] instruction [/INST] response
        prompt = f"[INST] {user_input} [/INST]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,       # Added for slightly better coherence
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return full_response.strip()