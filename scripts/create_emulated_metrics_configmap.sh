#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${MZ_TEST_NAMESPACE:-ksense}"
CONFIGMAP_NAME="${MZ_EMULATED_DATA_CONFIGMAP:-ksense-emulated-metrics-data}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KERNEL_CSV="${1:-${ROOT_DIR}/kernel_metrics.csv}"
HOTEL1_CSV="${2:-${ROOT_DIR}/hotel1_sa__hotel1_sa2.csv}"
HOTEL2_CSV="${3:-${ROOT_DIR}/hotel1_sn1_sa__hotel1_sn1_sa5.csv}"

if [[ ! -f "${KERNEL_CSV}" ]]; then
  echo "Missing file: ${KERNEL_CSV}"
  echo "Usage: $0 [kernel_metrics.csv] [hotel_csv_1] [hotel_csv_2]"
  exit 1
fi
if [[ ! -f "${HOTEL1_CSV}" ]]; then
  echo "Missing file: ${HOTEL1_CSV}"
  echo "Usage: $0 [kernel_metrics.csv] [hotel_csv_1] [hotel_csv_2]"
  exit 1
fi
if [[ ! -f "${HOTEL2_CSV}" ]]; then
  echo "Missing file: ${HOTEL2_CSV}"
  echo "Usage: $0 [kernel_metrics.csv] [hotel_csv_1] [hotel_csv_2]"
  exit 1
fi

kubectl -n "${NAMESPACE}" delete configmap "${CONFIGMAP_NAME}" --ignore-not-found >/dev/null 2>&1 || true

CMD=(kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP_NAME}" --from-file=kernel_metrics.csv="${KERNEL_CSV}")
CMD+=(--from-file=hotel1_sa__hotel1_sa2.csv="${HOTEL1_CSV}")
CMD+=(--from-file=hotel1_sn1_sa__hotel1_sn1_sa5.csv="${HOTEL2_CSV}")

"${CMD[@]}"

echo "ConfigMap ${CONFIGMAP_NAME} updated in namespace ${NAMESPACE}"
