## Kubernetes Deployment Guide

This folder contains Kubernetes manifests for the fuzzy admission controller stack
and the proactive resource-offer API.

### Quick production deploy (single file)

Use this when you already have a real usage metrics API.

1. Edit `MZ_USAGE_API_URL` in `ksense-resource-offer-deployment.yaml`.
2. Apply:

```bash
kubectl apply -f kubernetes/ksense-resource-offer-deployment.yaml
```

This single file creates:
- namespace `ksense`
- deployment `ksense-resource-offer-api`
- service `ksense-resource-offer-api` on port `8080`

Important:
- `MZ_CONTROLLER_REPLAY=false` is set for production so fuzzy reads live controller CSV.
- `MZ_CONTROLLER_CSV=/var/run/ksense/kernel_metrics.csv` expects a continuously updated file.
- The deployment mounts host path `/tmp/ksense` to `/var/run/ksense` (read-only).

### Access the sellable resources API

- From another pod in cluster:

```bash
curl http://ksense-resource-offer-api.ksense.svc:8080/resource_offer
```

- From your local machine:

```bash
kubectl -n ksense port-forward svc/ksense-resource-offer-api 8080:8080
curl http://127.0.0.1:8080/resource_offer
```

Useful endpoints:
- `GET /healthz`
- `GET /resource_offer`
- `GET /resource_offer_debug`
- `GET /resource_offer_now`

### Alternative production manifest

If you prefer a separate production file (without embedded namespace object):

```bash
kubectl apply -f kubernetes/resource-offer-prod.yaml
```

### Emulation test stack (optional)

Use this only for replaying hotel CSV data in-cluster.

```bash
scripts/create_emulated_metrics_configmap.sh
kubectl apply -f kubernetes/resource-offer-test-stack.yaml
kubectl apply -f kubernetes/resource-offer-smoke-job.yaml
```

### Full fuzzy scheduler/webhook stack

For the original collector + webhook + scheduler deployment:

```bash
scripts/setup_webhook_tls.sh
kubectl apply -k kubernetes/
```
