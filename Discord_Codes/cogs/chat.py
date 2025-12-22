import discord
from discord.ext import commands
import asyncio

class AIChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        # Ignore bot's own messages
        if message.author == self.bot.user:
            return

        # Don't respond if it starts with a command prefix (like !join)
        if message.content.startswith(self.bot.command_prefix):
            return

        # Access the AI Engine we attached in main.py
        if hasattr(self.bot, 'llm_engine'):
            async with message.channel.typing():
                # Run blocking code in a separate thread
                response = await asyncio.to_thread(
                    self.bot.llm_engine.generate_response, 
                    message.content
                )
                await message.channel.send(response)
        else:
            print("Error: LLM Engine not loaded.")

async def setup(bot):
    await bot.add_cog(AIChat(bot))