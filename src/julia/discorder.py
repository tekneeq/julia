# https://discord.com/developers/applications

import os
import discord
from discord import Intents
from dotenv import load_dotenv
import time


load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

intents = Intents.default()  # set more if you need them
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    channel = client.get_channel(CHANNEL_ID)
    await channel.send("Hello from my bot! 🤖")
    # await client.close()  # optional: exit after sending once


@client.event
async def on_message(msg):
    if msg.author.bot:
        return
    if msg.content == "!ping":
        await msg.reply("pong!")


client.run(TOKEN)
