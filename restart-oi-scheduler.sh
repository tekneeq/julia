#!/usr/bin/env bash
#
# Restart (or inspect) the OI snapshot scheduler inside the dashboard
# container.
#
# By default the scheduler now starts DETACHED, so it survives the shell
# that launched it. Console output goes to logs/oi-scheduler.out on the
# host; per-fire batch output to logs/oi-scheduler/YYYY-MM-DD.log and
# per-job output to logs/oi-batch/TICKER-DATE.log.
#
# Usage
#   ./restart-oi-scheduler.sh                restart, detached
#   ./restart-oi-scheduler.sh --now          restart + fire a batch immediately
#   ./restart-oi-scheduler.sh --run-once     fire ONE batch now and exit (no restart)
#   ./restart-oi-scheduler.sh --status       what ran, what's missing (no restart)
#   ./restart-oi-scheduler.sh --stop         stop the scheduler
#   ./restart-oi-scheduler.sh --dry-run      print the next 10 fire times
#   ./restart-oi-scheduler.sh --logs         follow the scheduler console log
#   ./restart-oi-scheduler.sh --foreground   run attached (previous behaviour)
#
# Combine any of the above with:
#   -t, --tickers SPY,QQQ     -a, --days-ahead 7
#   -i, --interval-min 15     -w, --workers 4
#
# Note: a detached `docker exec` does not survive a container restart, so
# run this again after ./restart.sh or a host reboot.

set -euo pipefail

cd "$(dirname "$0")"

CONTAINER="${OI_CONTAINER:-julia-dashboard}"
SCHED="scripts/oi_scheduler.py"
OUT_LOG="logs/oi-scheduler.out"

MODE="restart"
FOREGROUND=0
PASSTHRU=()

usage() { sed -n '2,/^[^#]/p' "$0" | sed 's/^#//; s/^ //' | sed '$d'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--now)          PASSTHRU+=(--now); shift ;;
        -r|--run-once)     MODE="run-once"; shift ;;
        -s|--status)       MODE="status"; shift ;;
        --stop)            MODE="stop"; shift ;;
        -d|--dry-run)      MODE="dry-run"; shift ;;
        -l|--logs)         MODE="logs"; shift ;;
        -f|--foreground)   FOREGROUND=1; shift ;;
        -t|--tickers)      PASSTHRU+=(--tickers "$2"); shift 2 ;;
        -a|--days-ahead)   PASSTHRU+=(--days-ahead "$2"); shift 2 ;;
        -i|--interval-min) PASSTHRU+=(--interval-min "$2"); shift 2 ;;
        -w|--workers)      PASSTHRU+=(--workers "$2"); shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *)                 PASSTHRU+=("$1"); shift ;;
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

# status / dry-run / stop / run-once are all short-lived and want their
# output on your terminal — no detaching, no pid juggling.
if [[ "$MODE" != "restart" ]]; then
    exec docker exec "$CONTAINER" uv run python "$SCHED" "--$MODE" "$@"
fi

if [[ $FOREGROUND -eq 1 ]]; then
    exec docker exec "${TTY[@]+"${TTY[@]}"}" "$CONTAINER" \
        uv run python "$SCHED" --replace "$@"
fi

# Re-quote so args survive the `sh -c` hop into the container.
QUOTED=""
for a in --replace "$@"; do QUOTED+=" $(printf '%q' "$a")"; done

# Drop recycled PIDs left on the host-mounted logs/ volume after a
# container recreate — the Python side also scrubs, but clear eagerly.
# Pid files are root-owned (written in-container); delete via docker exec.
docker exec "$CONTAINER" rm -f logs/oi-scheduler.pid

# ``setsid`` + ignore SIGHUP so the process survives ``docker exec -d``
# session teardown (plain ``exec uv run`` was dying right after deploy).
docker exec -d "$CONTAINER" sh -c \
    "mkdir -p logs && setsid uv run python $SCHED$QUOTED >>$OUT_LOG 2>&1 </dev/null"

# Confirm it actually came up rather than reporting success for a process
# that died on startup.
sleep 5
echo "── scheduler status ──────────────────────────────────────────────"
STATUS="$(docker exec "$CONTAINER" uv run python "$SCHED" --status "$@" || true)"
printf '%s\n' "$STATUS"
echo
echo "Console log: $OUT_LOG   (follow with: $0 --logs)"
if ! printf '%s\n' "$STATUS" | grep -q 'Process: ● RUNNING'; then
    echo "❌ scheduler failed to stay up — last log lines:" >&2
    docker exec "$CONTAINER" sh -c "tail -n 60 $OUT_LOG" 2>/dev/null || true
    exit 1
fi
