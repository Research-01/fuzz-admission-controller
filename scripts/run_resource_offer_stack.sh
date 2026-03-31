#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${MZ_RUN_DIR:-/tmp/ksense}"

mkdir -p "$RUN_DIR"

EMU_PID_FILE="$RUN_DIR/emulated_usage_api.pid"
OFFER_PID_FILE="$RUN_DIR/resource_offer_api.pid"
EMU_LOG="$RUN_DIR/emulated_usage_api.log"
OFFER_LOG="$RUN_DIR/resource_offer_api.log"

EMU_HOST="${MZ_EMULATED_USAGE_HOST:-127.0.0.1}"
EMU_PORT="${MZ_EMULATED_USAGE_PORT:-8090}"
EMU_TICK_S="${MZ_EMULATED_USAGE_TICK_S:-5}"
EMU_LOOP="${MZ_EMULATED_USAGE_LOOP:-true}"
EMU_CSVS_DEFAULT="$ROOT_DIR/hotel1_sa__hotel1_sa2.csv,$ROOT_DIR/hotel1_sn1_sa__hotel1_sn1_sa5.csv"
EMU_CSVS="${MZ_EMULATED_USAGE_CSVS:-$EMU_CSVS_DEFAULT}"

OFFER_HOST="${RESOURCE_OFFER_HOST:-127.0.0.1}"
OFFER_PORT="${RESOURCE_OFFER_PORT:-8080}"
USAGE_API_URL="${MZ_USAGE_API_URL:-http://$EMU_HOST:$EMU_PORT/usage/latest}"
USAGE_API_POLL_S="${MZ_USAGE_API_POLL_S:-5}"
USAGE_API_CSV="${MZ_USAGE_API_CSV:-$RUN_DIR/usage_api_metrics.csv}"
CONTROLLER_CSV="${MZ_CONTROLLER_CSV:-$ROOT_DIR/kernel_metrics.csv}"
OFFER_REFRESH_S="${MZ_OFFER_REFRESH_S:-40}"
PREDICT_WINDOW_S="${MZ_PREDICT_WINDOW_S:-300}"
RESOURCE_OFFER_CSV="${MZ_RESOURCE_OFFER_CSV:-$RUN_DIR/resource_offer.csv}"
FUZZY_SCORE_CSV="${FUZZY_SCORE_CSV:-$RUN_DIR/fuzzy_score.csv}"
PLOT_OUTPUT="${MZ_RESOURCE_OFFER_PLOT:-$RUN_DIR/resource_offer_behavior.png}"

is_pid_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

pid_from_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    cat "$path"
  fi
}

cleanup_stale_pid() {
  local path="$1"
  local pid
  pid="$(pid_from_file "$path")"
  if [[ -n "${pid:-}" ]] && ! is_pid_running "$pid"; then
    rm -f "$path"
  fi
}

wait_http_ok() {
  local url="$1"
  local timeout_s="${2:-20}"
  local i
  for ((i = 0; i < timeout_s; i++)); do
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_emulated() {
  cleanup_stale_pid "$EMU_PID_FILE"
  local pid
  pid="$(pid_from_file "$EMU_PID_FILE")"
  if [[ -n "${pid:-}" ]] && is_pid_running "$pid"; then
    echo "emulated-usage-api already running (pid=$pid)"
    return 0
  fi

  echo "starting emulated-usage-api on $EMU_HOST:$EMU_PORT"
  (
    cd "$ROOT_DIR"
    MZ_EMULATED_USAGE_HOST="$EMU_HOST" \
    MZ_EMULATED_USAGE_PORT="$EMU_PORT" \
    MZ_EMULATED_USAGE_TICK_S="$EMU_TICK_S" \
    MZ_EMULATED_USAGE_LOOP="$EMU_LOOP" \
    MZ_EMULATED_USAGE_CSVS="$EMU_CSVS" \
    setsid python3 -u emulated_usage_api.py >>"$EMU_LOG" 2>&1 < /dev/null &
    echo $! >"$EMU_PID_FILE"
  )

  if ! wait_http_ok "http://$EMU_HOST:$EMU_PORT/healthz" 30; then
    echo "failed to start emulated-usage-api; check $EMU_LOG"
    exit 1
  fi
}

start_offer() {
  cleanup_stale_pid "$OFFER_PID_FILE"
  local pid
  pid="$(pid_from_file "$OFFER_PID_FILE")"
  if [[ -n "${pid:-}" ]] && is_pid_running "$pid"; then
    echo "resource-offer-api already running (pid=$pid)"
    return 0
  fi

  echo "starting resource-offer-api on $OFFER_HOST:$OFFER_PORT"
  (
    cd "$ROOT_DIR"
    RESOURCE_OFFER_HOST="$OFFER_HOST" \
    RESOURCE_OFFER_PORT="$OFFER_PORT" \
    MZ_USAGE_API_URL="$USAGE_API_URL" \
    MZ_USAGE_API_POLL_S="$USAGE_API_POLL_S" \
    MZ_USAGE_API_CSV="$USAGE_API_CSV" \
    MZ_CONTROLLER_CSV="$CONTROLLER_CSV" \
    MZ_OFFER_REFRESH_S="$OFFER_REFRESH_S" \
    MZ_PREDICT_WINDOW_S="$PREDICT_WINDOW_S" \
    MZ_RESOURCE_OFFER_CSV="$RESOURCE_OFFER_CSV" \
    FUZZY_SCORE_CSV="$FUZZY_SCORE_CSV" \
    setsid python3 -u resource_offer_api.py >>"$OFFER_LOG" 2>&1 < /dev/null &
    echo $! >"$OFFER_PID_FILE"
  )

  if ! wait_http_ok "http://$OFFER_HOST:$OFFER_PORT/healthz" 30; then
    echo "failed to start resource-offer-api; check $OFFER_LOG"
    exit 1
  fi
}

stop_one() {
  local name="$1"
  local pid_file="$2"
  cleanup_stale_pid "$pid_file"
  local pid
  pid="$(pid_from_file "$pid_file")"
  if [[ -z "${pid:-}" ]]; then
    echo "$name not running"
    return 0
  fi
  if is_pid_running "$pid"; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if is_pid_running "$pid"; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    echo "stopped $name (pid=$pid)"
  fi
  rm -f "$pid_file"
}

status_one() {
  local name="$1"
  local pid_file="$2"
  local health_url="$3"
  cleanup_stale_pid "$pid_file"
  local pid
  pid="$(pid_from_file "$pid_file")"
  if [[ -z "${pid:-}" ]]; then
    echo "$name: stopped"
    return 0
  fi
  if ! is_pid_running "$pid"; then
    echo "$name: stale pid=$pid"
    return 0
  fi
  if curl -fsS --max-time 1 "$health_url" >/dev/null 2>&1; then
    echo "$name: running pid=$pid (healthy)"
  else
    echo "$name: running pid=$pid (health check failed)"
  fi
}

plot_now() {
  if [[ ! -f "$RESOURCE_OFFER_CSV" ]]; then
    echo "resource offer csv not found: $RESOURCE_OFFER_CSV"
    exit 1
  fi
  local fuzzy_arg=()
  if [[ -f "$FUZZY_SCORE_CSV" ]]; then
    fuzzy_arg=(--fuzzy-score "$FUZZY_SCORE_CSV")
  fi
  (
    cd "$ROOT_DIR"
    python3 scripts/plot_resource_offer_behavior.py \
      --input "$RESOURCE_OFFER_CSV" \
      "${fuzzy_arg[@]}" \
      --output "$PLOT_OUTPUT"
  )
  echo "plot written to $PLOT_OUTPUT"
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status|plot|tail}

Environment overrides:
  MZ_EMULATED_USAGE_HOST, MZ_EMULATED_USAGE_PORT, MZ_EMULATED_USAGE_CSVS, MZ_EMULATED_USAGE_TICK_S
  RESOURCE_OFFER_HOST, RESOURCE_OFFER_PORT, MZ_USAGE_API_URL, MZ_OFFER_REFRESH_S, MZ_CONTROLLER_CSV
  MZ_RUN_DIR, MZ_USAGE_API_CSV, MZ_RESOURCE_OFFER_CSV, FUZZY_SCORE_CSV, MZ_RESOURCE_OFFER_PLOT
EOF
}

cmd="${1:-}"
case "$cmd" in
  start)
    start_emulated
    start_offer
    status_one "emulated-usage-api" "$EMU_PID_FILE" "http://$EMU_HOST:$EMU_PORT/healthz"
    status_one "resource-offer-api" "$OFFER_PID_FILE" "http://$OFFER_HOST:$OFFER_PORT/healthz"
    echo "logs: $EMU_LOG | $OFFER_LOG"
    echo "offer endpoint: http://$OFFER_HOST:$OFFER_PORT/resource_offer"
    ;;
  stop)
    stop_one "resource-offer-api" "$OFFER_PID_FILE"
    stop_one "emulated-usage-api" "$EMU_PID_FILE"
    ;;
  restart)
    stop_one "resource-offer-api" "$OFFER_PID_FILE"
    stop_one "emulated-usage-api" "$EMU_PID_FILE"
    start_emulated
    start_offer
    status_one "emulated-usage-api" "$EMU_PID_FILE" "http://$EMU_HOST:$EMU_PORT/healthz"
    status_one "resource-offer-api" "$OFFER_PID_FILE" "http://$OFFER_HOST:$OFFER_PORT/healthz"
    ;;
  status)
    status_one "emulated-usage-api" "$EMU_PID_FILE" "http://$EMU_HOST:$EMU_PORT/healthz"
    status_one "resource-offer-api" "$OFFER_PID_FILE" "http://$OFFER_HOST:$OFFER_PORT/healthz"
    ;;
  plot)
    plot_now
    ;;
  tail)
    echo "--- emulated usage log ($EMU_LOG) ---"
    tail -n 50 "$EMU_LOG" || true
    echo "--- resource offer log ($OFFER_LOG) ---"
    tail -n 50 "$OFFER_LOG" || true
    ;;
  *)
    usage
    exit 1
    ;;
esac
