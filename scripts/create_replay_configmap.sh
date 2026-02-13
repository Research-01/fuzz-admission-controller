#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:-}"
NAMESPACE="${FUZZY_REPLAY_NAMESPACE:-ksense}"
CONFIGMAP_NAME="${FUZZY_REPLAY_CONFIGMAP:-fuzzy-replay-data}"

if [[ -z "${CSV_PATH}" ]]; then
  echo "Usage: $0 /path/to/kernel_metrics.csv"
  exit 1
fi

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "File not found: ${CSV_PATH}"
  exit 1
fi

kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP_NAME}" \
  --from-file=kernel_metrics.csv="${CSV_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "ConfigMap ${CONFIGMAP_NAME} updated in namespace ${NAMESPACE}"
