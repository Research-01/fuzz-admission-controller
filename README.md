## Fuzzy Admission Controller (kernel metrics based)

This repo runs a node-local kernel metrics collector, a fuzzy scoring webhook,
and a custom scheduler. The collector provides friction/energy signals; the
fuzzy controller combines those signals with CPU and PSI to compute a score.

Important behavior: the webhook always allows Pod CREATE and only records the
score/decision. The scheduler decides whether to bind the pod; if the score is
too high, the pod stays Pending.

### Collector-only (no webhook / no Docker / no Kubernetes)

If you only want kernel metrics + derived friction/energy + CPU/PSI in **one CSV**
directly on a node, run the collector from your terminal:

1) Install system deps (Ubuntu example):

```bash
sudo apt-get update
sudo apt-get install -y bpfcc-tools python3-bpfcc python3-pip linux-headers-$(uname -r)
```

2) Install Python deps:

```bash
python3 -m pip install -r requirements.collector.txt
```

3) Run (needs root/privileges for eBPF):

```bash
sudo mount -t tracefs nodev /sys/kernel/tracing 2>/dev/null || true
sudo mount -t debugfs none /sys/kernel/debug 2>/dev/null || true

sudo -E KSENSE_METRICS_CSV=/tmp/ksense/kernel_metrics.csv python3 collector_only.py
```

Output file (single CSV):
- `/tmp/ksense/kernel_metrics.csv` (override with `KSENSE_METRICS_CSV`)

CSV columns include:
- scheduler runnable latency stats (avg/p95/p99/max, counts)
- D-state time stats
- softirq NET_RX time stats
- `CPUUtil` and `PSI` (sampled from `/proc`)
- baseline state + derived `Friction`, `Direction`, `Energy` (+ energy internals)

Notes:
- The first row may have empty `CPUUtil`/`PSI` (delta-based sampling needs a previous sample).
- During warmup/baseline calibration, `Friction`/`Energy` fields are empty until the baseline freezes (default warmup is 600s).
- Watch it live: `tail -f /tmp/ksense/kernel_metrics.csv`

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
- `fuzzy_monitor.csv` (1s, written by the webhook controller)
- `fuzzy_score.csv` (written by the webhook controller)

Collector-only mode (no webhook) writes only:
- `kernel_metrics.csv`

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
