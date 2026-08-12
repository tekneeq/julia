#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

git pull

GIT_SHA="$(git rev-parse --short HEAD)"
# Committer timestamp (= merge time for merge commits on main).
GIT_COMMIT_TIME="$(git show -s --format=%cI HEAD)"

docker build \
    --build-arg "GIT_SHA=${GIT_SHA}" \
    --build-arg "GIT_COMMIT_TIME=${GIT_COMMIT_TIME}" \
    -t julia-dashboard:latest .
docker rm -f julia-dashboard
docker run -d --name julia-dashboard --restart unless-stopped \
    -p 8501:8501 \
    --memory=3g --memory-swap=3g \
    -v "$(pwd)/.options_cache:/app/.options_cache" \
    -v "$(pwd)/plots:/app/plots" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/.tokens:/root/.tokens" \
    --env-file .env \
    -e "JULIA_GIT_SHA=${GIT_SHA}" \
    -e "JULIA_GIT_COMMIT_TIME=${GIT_COMMIT_TIME}" \
    julia-dashboard:latest

echo "Started julia-dashboard at ${GIT_SHA} (${GIT_COMMIT_TIME})"
