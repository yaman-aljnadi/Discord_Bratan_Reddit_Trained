import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MistralEngine:
    def __init__(self):
        print("Loading Mistral-7B... this might take a minute.")
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            device_map="cuda", 
            trust_remote_code=True
        )
        print("Mistral Loaded successfully!")

    def generate_response(self, user_input):
        prompt = f"[INST] {user_input} [/INST]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256,   
                do_sample=True,       
                temperature=0.7,      
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if "[/INST]" in full_response:
            return full_response.split("[/INST]")[1].strip()
        
        return full_response.strip()