# import discord
# from discord.ext import commands

# class VoiceCommands(commands.Cog):
#     def __init__(self, bot):
#         self.bot = bot

#     @commands.command()
#     async def join(self, ctx):
#         """Joins the voice channel."""
#         if ctx.author.voice:
#             channel = ctx.author.voice.channel
#             if ctx.voice_client:
#                 await ctx.voice_client.move_to(channel)
#             else:
#                 await channel.connect()
#             await ctx.send(f"Joined **{channel.name}**!")
#         else:
#             await ctx.send("You need to be in a voice channel first.")

#     @commands.command(aliases=['kick'])
#     async def leave(self, ctx):
#         """Leaves the voice channel."""
#         if ctx.voice_client:
#             await ctx.voice_client.disconnect()
#             await ctx.send("Disconnected.")
#         else:
#             await ctx.send("I'm not connected to a voice channel.")

# # This function is required for main.py to load this file
# async def setup(bot):
#     await bot.add_cog(VoiceCommands(bot))


import discord
from discord.ext import commands
import os

class VoiceCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def join(self, ctx):
        """Joins the voice channel."""
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            if ctx.voice_client:
                await ctx.voice_client.move_to(channel)
            else:
                await channel.connect()
            await ctx.send(f"Joined **{channel.name}**!")
        else:
            await ctx.send("You need to be in a voice channel first.")

    @commands.command(aliases=['kick'])
    async def leave(self, ctx):
        """Leaves the voice channel."""
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
            await ctx.send("Disconnected.")
        else:
            await ctx.send("I'm not connected to a voice channel.")

    @commands.command()
    async def play(self, ctx, *, filename: str):
        """Plays a local .mp3 file."""
        # Auto-append .mp3 if the user just types the name
        if not filename.endswith('.mp3'):
            filename += '.mp3'
        
        # Check if the file actually exists in your bot's folder
        if not os.path.exists(filename):
            await ctx.send(f"❌ Error: Could not find the file `{filename}` in the bot's directory.")
            return

        # Auto-connect if the bot isn't in a voice channel
        if not ctx.voice_client:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                await channel.connect()
            else:
                await ctx.send("You need to be in a voice channel first so I know where to join.")
                return

        voice_client = ctx.voice_client

        # Stop currently playing audio before starting a new one
        if voice_client.is_playing():
            voice_client.stop()

        try:
            # Stream the local file
            source = discord.FFmpegPCMAudio(filename)
            voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)
            await ctx.send(f"🎵 Now playing: `{filename}`")
            
        except Exception as e:
            await ctx.send(f"❌ Error playing audio. Details: {e}")
            print(f"Audio Error: {e}")

async def setup(bot):
    await bot.add_cog(VoiceCommands(bot))