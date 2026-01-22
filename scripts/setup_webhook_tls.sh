#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${FUZZY_WEBHOOK_NAMESPACE:-ksense}"
SERVICE="${FUZZY_WEBHOOK_SERVICE:-ksense-fuzzy-webhook}"
CERT_DIR="${FUZZY_CERT_DIR:-./kubernetes/certs}"

mkdir -p "${CERT_DIR}"

CA_KEY="${CERT_DIR}/ca.key"
CA_CERT="${CERT_DIR}/ca.crt"
TLS_KEY="${CERT_DIR}/tls.key"
TLS_CERT="${CERT_DIR}/tls.crt"
CSR="${CERT_DIR}/tls.csr"

cat > "${CERT_DIR}/csr.conf" <<EOF
[req]
distinguished_name=req_distinguished_name
req_extensions=v3_req
prompt=no

[req_distinguished_name]
CN=${SERVICE}.${NAMESPACE}.svc

[v3_req]
keyUsage=keyEncipherment,dataEncipherment
extendedKeyUsage=serverAuth
subjectAltName=@alt_names

[alt_names]
DNS.1=${SERVICE}
DNS.2=${SERVICE}.${NAMESPACE}
DNS.3=${SERVICE}.${NAMESPACE}.svc
EOF

openssl genrsa -out "${CA_KEY}" 2048
openssl req -x509 -new -nodes -key "${CA_KEY}" -subj "/CN=${SERVICE}-ca" -days 3650 -out "${CA_CERT}"
openssl genrsa -out "${TLS_KEY}" 2048
openssl req -new -key "${TLS_KEY}" -out "${CSR}" -config "${CERT_DIR}/csr.conf"
openssl x509 -req -in "${CSR}" -CA "${CA_CERT}" -CAkey "${CA_KEY}" -CAcreateserial \
  -out "${TLS_CERT}" -days 3650 -extensions v3_req -extfile "${CERT_DIR}/csr.conf"

kubectl -n "${NAMESPACE}" create secret tls fuzzy-webhook-tls \
  --cert="${TLS_CERT}" --key="${TLS_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

CA_BUNDLE="$(base64 -w0 < "${CA_CERT}")"
kubectl patch validatingwebhookconfiguration ksense-fuzzy-webhook \
  --type='json' \
  -p='[{"op":"replace","path":"/webhooks/0/clientConfig/caBundle","value":"'"${CA_BUNDLE}"'"}]'

echo "TLS configured. Secret fuzzy-webhook-tls updated, caBundle patched."
