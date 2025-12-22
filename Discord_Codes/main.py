import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download

load_dotenv()
TOKEN = os.getenv("BRATAN_TOKEN")

# --- CONFIGURATION ---
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TTS_VOICE = "af_bella"  # Options: af_sarah, am_michael, am_adam, etc.
TTS_SPEED = 1.0

# --- 1. SETUP KOKORO TTS ---
print("Setting up Kokoro TTS...")

if not os.path.exists("kokoro-v0_19.onnx"):
    print("Downloading Kokoro model weights (first run only)...")
    hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v0_19.onnx", local_dir=".")

if not os.path.exists("voices.json"):
    print("Downloading Voice configs...")
    hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices.json", local_dir=".")

# Load Kokoro on CPU to save VRAM for Mistral
kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
print("Kokoro TTS Loaded!")


# --- 2. SETUP MISTRAL LLM ---
print("Loading Mistral-7B... this might take a minute.")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    trust_remote_code=True
)

print("Mistral Loaded successfully!")


# --- HELPER FUNCTIONS ---

def generate_mistral_response(user_input):
    prompt = f"[INST] {user_input} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,   
            do_sample=True,       
            temperature=0.7,      
            pad_token_id=tokenizer.eos_token_id
        )


    generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if "[/INST]" in full_response:
        response_text = full_response.split("[/INST]")[1].strip()
    else:
        response_text = full_response
        
    return response_text

def generate_tts_audio(text):
    """Generates audio from text and saves it to a temp file."""
    # Ensure text isn't empty or just whitespace
    if not text or not text.strip():
        return None
        
    # Kokoro generation
    samples, sample_rate = kokoro.create(
        text, 
        voice=TTS_VOICE, 
        speed=TTS_SPEED, 
        lang="en-us"
    )
    
    filename = "response_audio.wav"
    sf.write(filename, samples, sample_rate)
    return filename


# --- DISCORD BOT ---

intents = discord.Intents.default()
intents.message_content = True 

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'SUCCESS: Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    if not message.content.startswith(bot.command_prefix):
        
        async with message.channel.typing():
            # 1. Generate Text (Thread 1)
            response_text = await asyncio.to_thread(generate_mistral_response, message.content)
            print(f"LLM Response: {response_text}")

            # 2. Generate Audio (Thread 2)
            # We run this in a thread too so the bot doesn't freeze during synthesis
            audio_file_path = await asyncio.to_thread(generate_tts_audio, response_text)

            # 3. Send Text + Audio
            if audio_file_path:
                try:
                    file = discord.File(audio_file_path)
                    await message.channel.send(content=response_text, file=file)
                except Exception as e:
                    print(f"Error sending audio: {e}")
                    await message.channel.send(response_text)
            else:
                await message.channel.send(response_text)

if __name__ == "__main__":
    bot.run(TOKEN)