import os
import praw
from dotenv import load_dotenv


load_dotenv()

# https://www.reddit.com/prefs/apps
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLEINT_SECRET"),
    user_agent=f"redder:comment-scraper:v1.0 (by u/{os.getenv('REDDIT_USERNAME')})",
)

for submission in reddit.subreddit("wallstreetbets").hot(limit=10):
    print(submission.title)
