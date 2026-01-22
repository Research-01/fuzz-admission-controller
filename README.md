## Fuzzy Admission Controller (K-Sense based)

This repo builds a node-local fuzzy admission controller on top of K-Sense.
K-Sense provides kernel friction/energy signals; this controller uses those
signals plus CPU and PSI to decide admit/deny. K-Sense itself comes from the
original upstream repo (see references in the paper and source).

### What runs

- K-Sense collector (per-node DaemonSet) -> `/tmp/ksense/kernel_metrics.csv`
- Fuzzy webhook (sidecar in the same DaemonSet)
- Custom scheduler (Deployment) that calls `/score` on each node

### Deploy to Kubernetes

```bash
kubectl apply -k kubernetes/
scripts/setup_webhook_tls.sh
```

Check status:

```bash
kubectl get ds -n ksense
kubectl get pods -n ksense -o wide
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
- `POST /validate` (AdmissionReview)

### Logs + CSV outputs

Inside the DaemonSet pod:
- K-Sense: `/tmp/ksense/kernel_metrics.csv`
- Fuzzy inputs (1s): `/tmp/ksense/fuzzy_monitor.csv`
- Fuzzy scores: `/tmp/ksense/fuzzy_score.csv`

Example:

```bash
kubectl exec -n ksense <pod> -c fuzzy-webhook -- tail -n 5 /tmp/ksense/fuzzy_score.csv
```

### Offline plotting

Copy the CSVs to your machine and plot:

```bash
POD=$(kubectl get pods -n ksense -l app=ksense -o jsonpath='{.items[0].metadata.name}')
kubectl cp -n ksense ${POD}:/tmp/ksense/fuzzy_monitor.csv /tmp/fuzzy_monitor.csv
kubectl cp -n ksense ${POD}:/tmp/ksense/fuzzy_score.csv /tmp/fuzzy_score.csv

python3 scripts/realtime_fuzzy_plot.py \
  --inputs /tmp/fuzzy_monitor.csv \
  --scores /tmp/fuzzy_score.csv
```
