import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
TOKEN = os.getenv("BRATAN_TOKEN")

print("Loading Mistral-7B... this might take a minute.")

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 


model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    trust_remote_code=True
)

print("Mistral Loaded successfully!")


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
        print(f"Generated Response: {response_text}")
    else:
        print("Warning: [/INST] token not found in response.")
        response_text = full_response

    return response_text


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

@bot.command()
async def join(ctx):
    """Joins the voice channel the user is currently in."""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        # Check if bot is already in a voice channel
        if ctx.voice_client: 
            await ctx.voice_client.move_to(channel)
        else:
            await channel.connect()
        await ctx.send(f"Joined **{channel.name}**!")
    else:
        await ctx.send("You need to be in a voice channel first so I can join you.")

@bot.command(aliases=['kick']) # This allows !kick to work as an alias for !leave
async def leave(ctx):
    """Leaves the current voice channel."""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected from voice channel.")
    else:
        await ctx.send("I'm not connected to a voice channel.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    # If it's a regular chat message (not starting with prefix), treat as AI Chat
    # You might want to restrict this to specific channels or mentions later.
    if not message.content.startswith(bot.command_prefix):
        
        # UX: Trigger "Bot is typing..." while GPU works
        async with message.channel.typing():
            
            # CRITICAL: Run the blocking model generation in a separate thread
            # This prevents the bot from "freezing" while waiting for the AI
            response = await asyncio.to_thread(generate_mistral_response, message.content)
            
            await message.channel.send(response)

if __name__ == "__main__":
    bot.run(TOKEN)




