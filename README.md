## Fuzzy Admission Controller (kernel metrics based)

This repo runs a node-local kernel metrics collector, a fuzzy scoring webhook,
and a custom scheduler. The collector provides friction/energy signals; the
fuzzy controller combines those signals with CPU and PSI to compute a score.

Important behavior: the webhook always allows Pod CREATE and only records the
score/decision. The scheduler decides whether to bind the pod; if the score is
too high, the pod stays Pending.

### What runs

- Kernel metrics collector (per-node DaemonSet) -> shared metrics CSV
- Fuzzy webhook sidecar (same DaemonSet) -> `/score` + `/validate`
- Custom scheduler (Deployment) that calls `/score` and binds pods

### Build images

```bash
docker build -t your-registry/fuzzy-collector:latest -f Dockerfile .
docker build -t your-registry/fuzzy-scheduler:latest -f Dockerfile.scheduler .
docker push your-registry/fuzzy-collector:latest
docker push your-registry/fuzzy-scheduler:latest
```

### Deploy to Kubernetes

```bash
scripts/setup_webhook_tls.sh
kubectl apply -k kubernetes/
```

Check status:

```bash
kubectl get ds -n <collector-namespace>
kubectl get pods -n <collector-namespace> -o wide
kubectl get deploy -n <collector-namespace>
```

### Use the custom scheduler

Set this in your pod spec:

```yaml
spec:
  schedulerName: fuzzy-scheduler
```

### Webhook endpoints

- `GET /healthz`
- `GET /score` (node score JSON)
- `POST /validate` (AdmissionReview, always allows Pod CREATE)

### Proactive resource-offer API (localhost:8080)

This API exposes how much capacity can be safely sold, based on:
- Last 5 minutes of emulated usage history (`hotel*.csv` by default)
- A lightweight ARIMA-style forecast for CPU/PSI/RAM/storage usage
- Current fuzzy controller output (friction + energy + fuzzy score)

Run locally:

```bash
python3 resource_offer_api.py
```

This starts a proactive updater loop that refreshes the offer periodically
(default every 40s) and appends snapshots to `resource_offer.csv`.

Query:

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/resource_offer
```

### Mimic real input API (recommended for emulation)

If production will provide metrics from an API (not direct CSV), use this local flow:

1) Run the emulated usage API (replays `hotel*.csv`, updates every 5s):

```bash
python3 emulated_usage_api.py
```

When multiple CSV files are provided in `MZ_EMULATED_USAGE_CSVS`, playback is serial:
all rows of file 1, then file 2, then loop back.

2) Run resource-offer API with API mirroring enabled:

```bash
MZ_USAGE_API_URL=http://127.0.0.1:8090/usage/latest python3 resource_offer_api.py
```

This mirrors API payloads into a local CSV (`/tmp/ksense/usage_api_metrics.csv` by default),
and prediction uses that mirrored file.

### Keep the local stack running + plot behavior

Use the helper script to keep both APIs running in background and generate plots later:

```bash
scripts/run_resource_offer_stack.sh start
scripts/run_resource_offer_stack.sh status
curl http://127.0.0.1:8080/resource_offer_debug
```

At any later point, generate a plot from recorded CSVs:

```bash
scripts/run_resource_offer_stack.sh plot
```

This plot includes:
- Friction + Energy (from `fuzzy_score.csv`, if available)
- Fuzzy score + allow/deny decision
- Sellable CPU/RAM/Storage over time

Stop background services:

```bash
scripts/run_resource_offer_stack.sh stop
```

Response shape:

```json
{
  "cpu": 10.0,
  "ram": 66991.22,
  "GPU": 0.0,
  "storage": 34934.0
}
```

Static node totals (default, overridable by env):
- `MZ_TOTAL_CPU_CORES=256`
- `MZ_TOTAL_RAM_GB=2048` (2 TB)
- `MZ_TOTAL_STORAGE_GB=80078`
- `MZ_TOTAL_GPU=0`

Other useful env vars:
- `RESOURCE_OFFER_HOST` (default `127.0.0.1`)
- `RESOURCE_OFFER_PORT` (default `8080`)
- `MZ_USAGE_CSVS` (comma-separated usage CSV paths; default `hotel*.csv`)
- `MZ_USAGE_API_URL` (optional; if set, use API instead of direct usage CSV files)
- `MZ_USAGE_API_POLL_S` (default `5`)
- `MZ_USAGE_API_CSV` (default `/tmp/ksense/usage_api_metrics.csv`)
- `MZ_CONTROLLER_CSV` (controller input CSV; default `kernel_metrics.csv` if present)
- `MZ_PREDICT_WINDOW_S` (default `300`)
- `MZ_RESOURCE_OFFER_CSV` (default `/tmp/ksense/resource_offer.csv`)
- `MZ_OFFER_REFRESH_S` (default `40`)

Emulated usage API env vars:
- `MZ_EMULATED_USAGE_HOST` (default `127.0.0.1`)
- `MZ_EMULATED_USAGE_PORT` (default `8090`)
- `MZ_EMULATED_USAGE_CSVS` (default `hotel*.csv`)
- `MZ_EMULATED_USAGE_TICK_S` (default `5`)
- `MZ_EMULATED_USAGE_LOOP` (default `true`)

### Kubernetes smoke test for resource-offer path

This test runs fully in-cluster with two pods:
- `ksense-emulated-usage-api` (replays hotel CSV every 5s)
- `ksense-resource-offer-api` (mirrors usage API to local CSV and serves `/resource_offer`)

1) Load emulated CSVs into ConfigMap:

```bash
scripts/create_emulated_metrics_configmap.sh
```

2) Deploy test stack:

```bash
kubectl apply -f kubernetes/resource-offer-test-stack.yaml
kubectl -n ksense rollout status deploy/ksense-emulated-usage-api
kubectl -n ksense rollout status deploy/ksense-resource-offer-api
```

3) Run smoke job:

```bash
kubectl apply -f kubernetes/resource-offer-smoke-job.yaml
kubectl -n ksense logs -l app=ksense-resource-offer-smoke --tail=200
```

4) Manual checks:

```bash
kubectl -n ksense run tmp-curl --rm -it --restart=Never --image=curlimages/curl:8.7.1 -- \
  curl -fsS http://ksense-resource-offer-api:8080/resource_offer

POD=$(kubectl -n ksense get pod -l app=ksense-resource-offer-api -o jsonpath='{.items[0].metadata.name}')
kubectl -n ksense exec "$POD" -- tail -n 5 /tmp/ksense/usage_api_metrics.csv
kubectl -n ksense exec "$POD" -- tail -n 5 /tmp/ksense/resource_offer.csv
```

### Analyze accept/reject behavior (runtime)

`/resource_offer` serves the latest cached offer (refresh default 40s), so repeated calls
within the same refresh window can return identical values.

Use these debug endpoints:
- `GET /resource_offer_debug` -> latest snapshot with `decision`, `score`, `level`, `predicted`, `safety_multiplier`
- `GET /resource_offer_now` -> forces immediate recalculation and returns full snapshot
- `GET /reservations` -> active reserved resources not available for re-sale
- `POST /reserve` -> reserve part of current sellable offer (TTL-based)

Decision fields:
- `fuzzy_decision`: raw fuzzy output
- `decision`: final sellability decision after capacity guard
- `decision_reason`: `fuzzy_reject`, `fuzzy_allow`, or `capacity_guard`

Example via port-forward:

```bash
kubectl -n ksense port-forward svc/ksense-resource-offer-api 8080:8080
curl http://127.0.0.1:8080/resource_offer_debug
curl http://127.0.0.1:8080/resource_offer_now
curl http://127.0.0.1:8080/reservations
curl -X POST http://127.0.0.1:8080/reserve \
  -H "Content-Type: application/json" \
  -d '{"cpu":1,"ram":2,"storage":10,"ttl_s":60,"owner":"buyer-a"}'
```

Plot decision and sellable-resource trends from CSV:

```bash
POD=$(kubectl -n ksense get pod -l app=ksense-resource-offer-api -o jsonpath='{.items[0].metadata.name}')
kubectl -n ksense cp "${POD}:/tmp/ksense/resource_offer.csv" /tmp/resource_offer.csv
python3 scripts/plot_resource_offer_behavior.py --input /tmp/resource_offer.csv --output /tmp/resource_offer_behavior.png
```

### Decision logic (controller)

Score bands:
- High: score >= 70 -> deny (scheduler skips binding)
- Medium: 45..69 -> dynamic hysteresis
- Low: score < 45 -> allow

Medium bands are controlled by:
- `FUZZY_MEDIUM_LOW_UPPER` (default 55)
- `FUZZY_MEDIUM_MID_UPPER` (default 60)
- `FUZZY_MEDIUM_BAD_THRESHOLD` (default 2)

### Scheduler rate limits

Scheduler settings are configured via env vars (see `kubernetes/fuzzy-scheduler.yaml`):
- `FUZZY_SCHEDULER_POLL_S` (default 10)
- `FUZZY_SCHEDULER_MAX_PER_CYCLE` (default 1)
- `FUZZY_SCHEDULER_PLACEMENT_DELAY_S` (default 5)

### Webhook scope

The webhook only evaluates Pod CREATE in a specific namespace allowlist by
default (see the validating webhook manifest).

### Logs + CSV outputs

Inside the DaemonSet pod (shared data dir):
- `kernel_metrics.csv`
- `fuzzy_monitor.csv` (1s)
- `fuzzy_score.csv`

Example:

```bash
kubectl exec -n <collector-namespace> <pod> -c fuzzy-webhook -- tail -n 5 /tmp/<data-dir>/fuzzy_score.csv
```

### Test pods

Quick pause pods (9) using the custom scheduler:

```bash
kubectl apply -f kubernetes/fuzzy-test-pods.yaml
```

Lifecycle test script (feedback-inference-2..11 with NodePort services):

```bash
python3 src/<controller>/test.py --delete-services
```

### Offline plotting

Copy the CSVs to your machine and plot:

```bash
POD=$(kubectl get pods -n <collector-namespace> -l app=<collector-label> -o jsonpath='{.items[0].metadata.name}')
kubectl cp -n <collector-namespace> ${POD}:/tmp/<data-dir>/fuzzy_monitor.csv /tmp/fuzzy_monitor.csv
kubectl cp -n <collector-namespace> ${POD}:/tmp/<data-dir>/fuzzy_score.csv /tmp/fuzzy_score.csv

python3 scripts/realtime_fuzzy_plot.py \
  --inputs /tmp/fuzzy_monitor.csv \
  --scores /tmp/fuzzy_score.csv
```

### Offline replay (test controller behavior on a captured CSV)

If you have a captured `kernel_metrics.csv` (e.g., copied from another server) and want
to see what the controller would decide for each row:

```bash
python3 scripts/replay_controller_on_csv.py --input /path/to/kernel_metrics.csv --tail 10000 --print-every 100
```

This writes `/tmp/ksense/fuzzy_replay.csv` by default.

#### Run replay inside Kubernetes (pod/job)

If you want the replay to run **inside the cluster**, use the Job manifest:

```bash
scripts/create_replay_configmap.sh /path/to/kernel_metrics.csv
kubectl apply -f kubernetes/fuzzy-replay-job.yaml
kubectl logs -n ksense -l app=fuzzy-replay -f
```

The Job writes `/tmp/ksense/fuzzy_replay.csv` in the pod.
