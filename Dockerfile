# Streamlit dashboard + lia CLI in a Debian 12 container (glibc 2.36).
# Sidesteps Amazon Linux 2's glibc 2.26 constraints entirely — every
# native-code wheel (pyarrow, numpy, scipy, etc.) resolves cleanly.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Pin the container to US Eastern time so `oi_scheduler.py`'s market-hour
# logic (MARKET_OPEN=09:30, MARKET_CLOSE=16:00, POST_CLOSE=16:05) means
# NYSE hours, not UTC. Any host TZ can still be overridden at runtime
# with `docker run -e TZ=...`.
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=America/New_York

# Layer 1: install deps only. Cached unless pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Layer 2: project source + local install.
COPY src ./src
COPY scripts ./scripts
COPY README ./
COPY README.md ./
RUN uv sync --frozen

# Persistent state — mount host paths here so the DB, PNGs, and logs
# survive container restarts.
VOLUME ["/app/.options_cache", "/app/plots", "/app/logs"]

EXPOSE 8501

# Baked in by restart.sh / CI so the live dashboard can show which
# commit it's serving (``.git`` is not copied into the image).
ARG GIT_SHA=unknown
ARG GIT_COMMIT_TIME=unknown
ENV JULIA_GIT_SHA=$GIT_SHA \
    JULIA_GIT_COMMIT_TIME=$GIT_COMMIT_TIME

ENV UV_NATIVE_TLS=true \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["uv", "run", "streamlit", "run", "scripts/oi_dashboard_app.py"]
