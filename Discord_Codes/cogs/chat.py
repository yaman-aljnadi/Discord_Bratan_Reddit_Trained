import discord
from discord.ext import commands
import asyncio
import os
# Import the advanced TTS library
from TTS.api import TTS

class AIChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        print("⏳ Loading XTTS v2 Model... (This uses GPU)")
        
        # Load the model specifically to the GPU ('cuda')
        # This will download the model on the first run.
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        
        print("✅ XTTS Model Loaded! Ready to speak.")

    def generate_voice_file(self, text):
        """
        Uses XTTS to generate high-quality audio.
        Requires a 'voice_sample.wav' file in your project folder to clone the voice.
        """
        # xtts_v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
        self.tts.tts_to_file(
            text=text, 
            speaker_wav="voice_sample.wav",  # The file it clones the voice from
            language="en", 
            file_path="tts_output.wav"
        )

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.content.startswith(self.bot.command_prefix):
            return

        if hasattr(self.bot, 'llm_engine'):
            async with message.channel.typing():
                # 1. Generate Text (LLM)
                response = await asyncio.to_thread(
                    self.bot.llm_engine.generate_response, 
                    message.content
                )
                
                # Send text response
                await message.channel.send(response)

                # 2. Generate Voice (TTS)
                if message.guild.voice_client and message.guild.voice_client.is_connected():
                    voice_client = message.guild.voice_client
                    
                    if voice_client.is_playing():
                        voice_client.stop()

                    try:
                        # Ensure the reference file exists before trying to clone
                        if not os.path.exists("voice_sample.wav"):
                            print("Error: 'voice_sample.wav' not found! Please add a voice sample.")
                            return

                        # Run XTTS in a separate thread (it's heavy!)
                        await asyncio.to_thread(self.generate_voice_file, response)

                        # XTTS outputs .wav, so we play that
                        source = discord.FFmpegPCMAudio("tts_output.wav")
                        voice_client.play(source)
                    
                    except Exception as e:
                        print(f"Error playing voice: {e}")
                        # Optional: Send error to chat for debugging
                        # await message.channel.send(f"Voice Error: {e}")

        else:
            print("Error: LLM Engine not loaded.")

async def setup(bot):
    await bot.add_cog(AIChat(bot))