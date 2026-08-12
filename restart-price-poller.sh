#!/usr/bin/env bash
#
# Restart (or inspect) the live price poller inside the dashboard
# container. The poller feeds the "Today's price action" chart with a
# tick every few seconds — independent of the 30-minute OI scheduler.
#
# The poller starts DETACHED, so it survives the shell that launched it.
# Console output goes to logs/price-poller.out on the host.
#
# Usage
#   ./restart-price-poller.sh                restart, detached
#   ./restart-price-poller.sh --status       is it up? tick coverage today
#   ./restart-price-poller.sh --once         poll once now (auth smoke test)
#   ./restart-price-poller.sh --stop         stop the poller
#   ./restart-price-poller.sh --logs         follow the poller console log
#   ./restart-price-poller.sh --foreground   run attached
#
# Combine any of the above with:
#   -t, --tickers SPY,QQQ     -i, --interval-sec 10
#
# Note: a detached `docker exec` does not survive a container restart, so
# run this again after ./restart.sh or a host reboot.

set -euo pipefail

cd "$(dirname "$0")"

CONTAINER="${OI_CONTAINER:-julia-dashboard}"
POLLER="scripts/price_poller.py"
OUT_LOG="logs/price-poller.out"

MODE="restart"
FOREGROUND=0
PASSTHRU=()

usage() { sed -n '2,/^[^#]/p' "$0" | sed 's/^#//; s/^ //' | sed '$d'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--status)        MODE="status"; shift ;;
        -o|--once)          MODE="once"; shift ;;
        --stop)             MODE="stop"; shift ;;
        -l|--logs)          MODE="logs"; shift ;;
        -f|--foreground)    FOREGROUND=1; shift ;;
        -t|--tickers)       PASSTHRU+=(--tickers "$2"); shift 2 ;;
        -i|--interval-sec)  PASSTHRU+=(--interval-sec "$2"); shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *)                  PASSTHRU+=("$1"); shift ;;
    esac
done

# Move the collected flags into "$@" — `"${arr[@]}"` on an empty array
# trips `set -u` on bash 3.2 (still the default on macOS).
set -- "${PASSTHRU[@]+"${PASSTHRU[@]}"}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "❌ container '$CONTAINER' is not running. Start it with ./restart.sh" >&2
    exit 1
fi

TTY=()
[ -t 0 ] && TTY=(-it)

if [[ "$MODE" == "logs" ]]; then
    exec docker exec "${TTY[@]+"${TTY[@]}"}" "$CONTAINER" \
        tail -n 100 -f "$OUT_LOG"
fi

# status / once / stop are short-lived and want their output on your
# terminal — no detaching, no pid juggling.
if [[ "$MODE" != "restart" ]]; then
    exec docker exec "$CONTAINER" uv run python "$POLLER" "--$MODE" "$@"
fi

if [[ $FOREGROUND -eq 1 ]]; then
    exec docker exec "${TTY[@]+"${TTY[@]}"}" "$CONTAINER" \
        uv run python "$POLLER" --replace "$@"
fi

# Re-quote so args survive the `sh -c` hop into the container.
QUOTED=""
for a in --replace "$@"; do QUOTED+=" $(printf '%q' "$a")"; done

# Pid files are root-owned (written in-container); delete via docker exec.
docker exec "$CONTAINER" rm -f logs/price-poller.pid

docker exec -d "$CONTAINER" sh -c \
    "mkdir -p logs && setsid uv run python $POLLER$QUOTED >>$OUT_LOG 2>&1 </dev/null"

# Confirm it actually came up rather than reporting success for a process
# that died on startup.
sleep 5
echo "── price poller status ───────────────────────────────────────────"
STATUS="$(docker exec "$CONTAINER" uv run python "$POLLER" --status "$@" || true)"
printf '%s\n' "$STATUS"
echo
echo "Console log: $OUT_LOG   (follow with: $0 --logs)"
if ! printf '%s\n' "$STATUS" | grep -q 'Process: ● RUNNING'; then
    echo "❌ price poller failed to stay up — last log lines:" >&2
    docker exec "$CONTAINER" sh -c "tail -n 60 $OUT_LOG" 2>/dev/null || true
    exit 1
fi
