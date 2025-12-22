import discord
from discord.ext import commands
import asyncio
from gtts import gTTS
import os

class AIChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    def create_voice_file(self, text):
        """
        Converts text to speech and saves it to a file.
        This is a blocking operation, so we run it in a thread later.
        """
        tts = gTTS(text=text, lang='ru')
        # Save to a generic file. This will be overwritten every time.
        tts.save("tts_output.mp3")

    @commands.Cog.listener()
    async def on_message(self, message):
        # Ignore bot's own messages
        if message.author == self.bot.user:
            return

        # Don't respond if it starts with a command prefix
        if message.content.startswith(self.bot.command_prefix):
            return

        # Check if the LLM engine is loaded
        if hasattr(self.bot, 'llm_engine'):
            async with message.channel.typing():
                # 1. Generate the Text Response (Blocking Code -> Thread)
                response = await asyncio.to_thread(
                    self.bot.llm_engine.generate_response, 
                    message.content
                )
                
                # 2. Send the Text Response to Discord
                await message.channel.send(response)

                # 3. Handle Voice Output
                # Check if the bot is in a voice channel in this guild
                if message.guild.voice_client and message.guild.voice_client.is_connected():
                    voice_client = message.guild.voice_client

                    if voice_client.is_playing():
                        voice_client.stop()

                    try:
                        await asyncio.to_thread(self.create_voice_file, response)

                        source = discord.FFmpegPCMAudio("tts_output.mp3")
                        voice_client.play(source)
                    
                    except Exception as e:
                        print(f"Error playing voice: {e}")
                        await message.channel.send("I tried to speak, but something went wrong with the audio.")
        else:
            print("Error: LLM Engine not loaded.")

async def setup(bot):
    await bot.add_cog(AIChat(bot))