#!/usr/bin/env sh
set -eu

MODE="${APP_MODE:-resource-offer}"

if [ "${MODE}" = "resource-offer" ]; then
  export RESOURCE_OFFER_HOST="${RESOURCE_OFFER_HOST:-0.0.0.0}"
  exec python3 /app/resource_offer_api.py
fi

if [ "${MODE}" = "emulated-usage" ]; then
  export MZ_EMULATED_USAGE_HOST="${MZ_EMULATED_USAGE_HOST:-0.0.0.0}"
  exec python3 /app/emulated_usage_api.py
fi

echo "unsupported APP_MODE: ${MODE}" >&2
echo "supported modes: resource-offer, emulated-usage" >&2
exit 1
