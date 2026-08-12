#!/usr/bin/env bash
#
# Pull latest code, rebuild the dashboard container, then re-attach the
# detached in-container services (OI scheduler + price poller).
#
# Order matters: ``restart.sh`` replaces the container, which kills any
# ``docker exec -d`` processes — so the scheduler and poller must be
# started *after* the container comes back up.
#
# Usage (on the EC2 host, from the repo root):
#   ./deploy.sh
#
# Called automatically by .github/workflows/deploy-ec2.yml on pushes to
# main (via SSH). Safe to run by hand any time.

set -euo pipefail

cd "$(dirname "$0")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "=== deploy start (cwd=$(pwd), rev=$(git rev-parse --short HEAD 2>/dev/null || echo '?')) ==="

log "1/4  rebuild dashboard container (git pull + docker build/run)"
./restart.sh

# Give Streamlit a moment to bind :8501 before we pile on more processes.
sleep 3
if ! docker ps --format '{{.Names}}' | grep -qx julia-dashboard; then
    log "ERROR: julia-dashboard container is not running after restart.sh"
    exit 1
fi

# PID files live on the host-mounted logs/ volume and survive container
# recreates — clear them so we never SIGTERM a recycled PID (e.g. Streamlit).
# They are usually root-owned (written inside the container), so delete
# via docker exec rather than host ``rm`` (which Permission-denies).
docker exec julia-dashboard rm -f \
    logs/oi-scheduler.pid logs/price-poller.pid

log "2/4  restart OI scheduler (detached)"
./restart-oi-scheduler.sh

log "3/4  restart price poller (detached)"
./restart-price-poller.sh

log "4/4  one-off batch to seed the current expiration window"
# Blocks until the batch finishes so 08-17 / 08-18 etc. get a snapshot
# immediately instead of waiting for the next :00/:30 slot.
./restart-oi-scheduler.sh --run-once

log "=== deploy done (rev=$(git rev-parse --short HEAD)) ==="
./restart-oi-scheduler.sh --status || true
./restart-price-poller.sh --status || true
