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
