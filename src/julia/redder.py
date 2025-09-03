import os
import praw
from dotenv import load_dotenv
import time


load_dotenv()

# https://www.reddit.com/prefs/apps
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=f"redder:comment-scraper:v1.0 (by u/{os.getenv('REDDIT_USERNAME')})",
)

username = "wsbapp"  # replace with any Reddit username
user = reddit.redditor(username)

submission_id = None
latest_submission = None
print(f"Latest threads by u/{username}:")
for idx, submission in enumerate(
    user.submissions.new(limit=1)
):  # limit=10 newest threads

    print("-" * 50)
    print(f"Index: {idx}")
    print(f"Title: {submission.title}")
    print(f"Subreddit: {submission.subreddit.display_name}")
    print(f"URL: {submission.url}")
    print(f"Score: {submission.score}")
    print(f"Permalink: https://reddit.com{submission.permalink}")
    submission_id = submission.id
    latest_submission = submission


# --- Tracking printed comments ---
printed_comment_ids = set()

comment_idx = 0
time_sleep = 0
while True:
    if time_sleep % 5 == 0:
        latest_submission.comments.replace_more(limit=0)
        for comment in latest_submission.comments.list():

            if comment.id in printed_comment_ids:
                continue

            printed_comment_ids.add(comment.id)
            comment_idx += 1
            print(f"[{comment_idx}]")
            print(f"[{comment.author}] {comment.body}")
            print("-" * 50)

    time.sleep(1)
    time_sleep += 1
    print(f"Time sleep: {time_sleep}")
