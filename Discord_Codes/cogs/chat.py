import discord
from discord.ext import commands
import asyncio
import os
import wave
# IMPORT PIPER DIRECTLY
from piper import PiperVoice

class AIChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        
        # --- CONFIGURATION ---
        # Point this to the .onnx file you downloaded in Step 2
        # If it is in the same folder, just use the filename 
        self.model_path = "./piper_tts/en_US-joe-medium.onnx"
        self.output_file = "tts_output.wav"
        
        # Pre-load the voice model into memory (Much faster!)
        print(f"Loading Piper Voice Model: {self.model_path}...")
        
        # We assume CUDA (GPU) since you are on a DGX
        # If this fails, set use_cuda=False
        try:
            self.voice = PiperVoice.load(self.model_path, use_cuda=True)
            print("Piper Voice loaded on GPU!")
        except Exception as e:
            print(f"GPU Load failed ({e}), falling back to CPU...")
            self.voice = PiperVoice.load(self.model_path, use_cuda=False)

    def create_voice_file(self, text):
        """
        Uses the loaded PiperVoice object to generate audio directly.
        """
        with wave.open(self.output_file, "wb") as wav_file:
            # synthesize_wav writes directly to the file object
            self.voice.synthesize_wav(text, wav_file)

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.content.startswith(self.bot.command_prefix):
            return

        if hasattr(self.bot, 'llm_engine'):
            async with message.channel.typing():
                # 1. Generate Text
                response = await asyncio.to_thread(
                    self.bot.llm_engine.generate_response, 
                    message.content
                )
                
                await message.channel.send(response)

                # 2. Handle Voice
                if message.guild.voice_client and message.guild.voice_client.is_connected():
                    voice_client = message.guild.voice_client

                    if voice_client.is_playing():
                        voice_client.stop()

                    try:
                        # 3. Generate Audio
                        # We still run this in a thread to avoid blocking the bot loop
                        await asyncio.to_thread(self.create_voice_file, response)

                        # 4. Play Audio
                        if os.path.exists(self.output_file):
                            source = discord.FFmpegPCMAudio(self.output_file)
                            voice_client.play(source)
                        else:
                            print("Error: Audio file was not generated.")
                    
                    except Exception as e:
                        print(f"Error playing voice: {e}")
                        await message.channel.send("Audio error.")
        else:
            print("Error: LLM Engine not loaded.")

async def setup(bot):
    await bot.add_cog(AIChat(bot))