import os
import praw
from dotenv import load_dotenv
import time


load_dotenv()

# https://www.reddit.com/prefs/apps
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLEINT_SECRET"),
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

# --- Continuous loop ---
while True:
    # Expand any "MoreComments" placeholders
    latest_submission.comments.replace_more(limit=0)

    # Loop through all comments
    for comment in latest_submission.comments.list():
        if comment.id not in printed_comment_ids:
            printed_comment_ids.add(comment.id)  # Mark as printed

            # Print new comment
            print(f"[{comment.author}] {comment.body}")
            print("-" * 50)

    # Wait a bit before checking again (to avoid hitting API rate limits)
    time.sleep(10)
# Daily Discussion Thread for September 03, 2025
# Weekend Discussion Thread for the Weekend of October 18, 2024
