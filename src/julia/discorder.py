# https://discord.com/developers/applications

import os
import discord
from discord import Intents
from dotenv import load_dotenv
import time
import praw


load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

intents = Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


###
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=f"redder:comment-scraper:v1.0 (by u/{os.getenv('REDDIT_USERNAME')})",
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
)

username = "wsbapp"  # replace with any Reddit username
user = reddit.redditor(username)

submission_id = None
latest_submission = None
print(f"Latest threads by u/{username}:")
for idx, submission in enumerate(user.submissions.new(limit=1)):

    submission_id = submission.id
    latest_submission = submission
###


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

    if msg.content == "!rpost":

        latest_submission.reply("This is my comment posted via PRAW 🤖")
        await msg.reply("pong!")


client.run(TOKEN)
