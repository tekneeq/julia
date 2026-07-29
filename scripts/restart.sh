#!/bin/bash
git pull
docker build -t julia-dashboard:latest .
docker rm -f julia-dashboard
docker run -d --name julia-dashboard --restart unless-stopped \
    -p 8501:8501 \
    --memory=3g --memory-swap=3g \
    -v $(pwd)/.options_cache:/app/.options_cache \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/.tokens:/root/.tokens \
    --env-file .env \
    julia-dashboard:latest