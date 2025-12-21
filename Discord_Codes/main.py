# Starting Steps of the Bratan Bot
import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
load_dotenv()

TOKEN = os.getenv("BRATAN_TOKEN")

print(TOKEN)



import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True 


bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f'SUCCESS: Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')


@bot.command()
async def ping(ctx):
    """Responds to !ping with Pong!"""
    await ctx.send('Pong!')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)


    if message.content.lower() == "hello":
        await message.channel.send("Hello there! I am ready to be an AI.")

bot.run(TOKEN)