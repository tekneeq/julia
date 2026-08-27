#!/usr/bin/env bash
#
# Restart (or inspect) the Julia Discord stock bot inside the dashboard
# container. Answers !price / !quote / !info / !sectors / !search etc.
#
# The bot starts DETACHED, so it survives the shell that launched it.
# Console output goes to logs/discord-bot.out on the host.
#
# Usage
#   ./restart-discord-bot.sh                restart, detached
#   ./restart-discord-bot.sh --status       is it up?
#   ./restart-discord-bot.sh --stop         stop the bot
#   ./restart-discord-bot.sh --logs         follow the bot console log
#   ./restart-discord-bot.sh --foreground   run attached
#
# Requires DISCORD_BOT_TOKEN in the host .env (loaded into the container
# via docker --env-file). If the token is missing, this script exits 0
# with a skip message so ``deploy.sh`` can keep going on hosts that have
# not configured Discord yet.
#
# Note: a detached `docker exec` does not survive a container restart, so
# run this again after ./restart.sh or a host reboot (deploy.sh does).

set -euo pipefail

cd "$(dirname "$0")"

CONTAINER="${OI_CONTAINER:-julia-dashboard}"
BOT="scripts/discord_bot.py"
OUT_LOG="logs/discord-bot.out"

MODE="restart"
FOREGROUND=0

usage() { sed -n '2,/^[^#]/p' "$0" | sed 's/^#//; s/^ //' | sed '$d'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--status)        MODE="status"; shift ;;
        --stop)             MODE="stop"; shift ;;
        -l|--logs)          MODE="logs"; shift ;;
        -f|--foreground)    FOREGROUND=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "❌ container '$CONTAINER' is not running. Start it with ./restart.sh" >&2
    exit 1
fi

# Soft-skip when Discord isn't configured — don't fail deploy.
TOKEN_SET="$(docker exec "$CONTAINER" sh -c \
    'if [ -n "${DISCORD_BOT_TOKEN:-}" ]; then echo yes; else echo no; fi')"
if [[ "$TOKEN_SET" != "yes" ]]; then
    echo "⏭  DISCORD_BOT_TOKEN unset in container — skipping Discord bot."
    echo "   Add DISCORD_BOT_TOKEN (and optional DISCORD_CHANNEL_ID) to .env,"
    echo "   recreate the container (./restart.sh), then re-run $0."
    exit 0
fi

TTY=()
[ -t 0 ] && TTY=(-it)

if [[ "$MODE" == "logs" ]]; then
    exec docker exec "${TTY[@]+"${TTY[@]}"}" "$CONTAINER" \
        tail -n 100 -f "$OUT_LOG"
fi

if [[ "$MODE" != "restart" ]]; then
    exec docker exec "$CONTAINER" uv run python "$BOT" "--$MODE"
fi

if [[ $FOREGROUND -eq 1 ]]; then
    exec docker exec "${TTY[@]+"${TTY[@]}"}" "$CONTAINER" \
        uv run python "$BOT" --replace
fi

docker exec "$CONTAINER" rm -f logs/discord-bot.pid

docker exec -d "$CONTAINER" sh -c \
    "mkdir -p logs && setsid uv run python $BOT --replace >>$OUT_LOG 2>&1 </dev/null"

sleep 5
echo "── discord bot status ────────────────────────────────────────────"
STATUS="$(docker exec "$CONTAINER" uv run python "$BOT" --status || true)"
printf '%s\n' "$STATUS"
echo
echo "Console log: $OUT_LOG   (follow with: $0 --logs)"
if ! printf '%s\n' "$STATUS" | grep -q 'Process: ● RUNNING'; then
    echo "❌ discord bot failed to stay up — last log lines:" >&2
    docker exec "$CONTAINER" sh -c "tail -n 60 $OUT_LOG" 2>/dev/null || true
    exit 1
fi
