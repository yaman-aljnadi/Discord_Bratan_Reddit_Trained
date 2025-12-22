import discord
from discord.ext import commands

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

# This function is required for main.py to load this file
async def setup(bot):
    await bot.add_cog(VoiceCommands(bot))