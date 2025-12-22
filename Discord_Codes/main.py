import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
from services.llm_engine import MistralEngine

# 1. Setup
load_dotenv()
TOKEN = os.getenv("BRATAN_TOKEN")

intents = discord.Intents.default()
intents.message_content = True 

bot = commands.Bot(command_prefix="!", intents=intents)

# 2. Attach the Brain to the Bot
# We do this here so the model is loaded once and accessible everywhere
bot.llm_engine = MistralEngine()

@bot.event
async def on_ready():
    print(f'SUCCESS: Logged in as {bot.user} (ID: {bot.user.id})')

# 3. Load Cogs and Start
async def main():
    # Load the files from the cogs folder
    await bot.load_extension("cogs.voice")
    await bot.load_extension("cogs.chat")
    
    # Start the bot
    await bot.start(TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle manual stop (Ctrl+C) gracefully
        print("Bot stopped by user.")