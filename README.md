# K-Sense: A Non-Invasive eBPF Framework for QoS Inference

This repository contains the reference implementation for K-Sense, a non-invasive
eBPF-based framework that infers QoS degradation from kernel behavior by computing
a covariance-aware friction signal (Mahalanobis distance) plus a directional sign
and an adaptive energy signal.

Paper status: This work is under review for IEEE ICFE 2026.

## License

All rights reserved until publication; we will update the license after the paper is accepted.

## Contact

For collaboration, contact:
anon@users.noreply.github.com
anon@users.noreply.github.com
anon@users.noreply.github.com

## Layout

- `src/ksense/` Python package (BPF program, helpers, energy model, main loop)
- `main.py` Entry point script
- `scripts/monitor_cpu_psi.py` CPU + PSI monitoring helper script (CSV only, no plotting)
- `scripts/step_response_prober.py` Step-response load generator for 3 apps (per-second CSV). Uses DeathStarBench and a sentiment-analysis app:
  - https://github.com/example/DeathStarBench
  - https://github.com/example/CustomerFeedback
- `scripts/latency_p99_prober.py` Parallel P99 latency prober for 3 apps (per-interval CSV). Uses the same app sources:
  - https://github.com/example/DeathStarBench
  - https://github.com/example/CustomerFeedback
- `kubernetes/` Kubernetes manifests (namespace, DaemonSet, kustomization)
- `node_collector/` Node-level syscall + kernel signal collector (eBPF) used to
  compute correlations and select which kernel syscalls/tracepoints best track latency

## Run

There are two separate components:
1) `K-Sense` main collector (friction/energy signals)
2) `node_collector` (kernel event correlation to select signals)

K-Sense main collector:

```bash
sudo -E python3 main.py
```


Node-level correlation collector:

```bash
sudo -E python3 node_collector/main.py
```

Scripts:

```bash
python3 scripts/monitor_cpu_psi.py
python3 scripts/step_response_prober.py
python3 scripts/latency_p99_prober.py
```


## Docker

K-Sense requires privileged access for eBPF. Example:

```bash
docker build -t ksense .
docker run --rm --privileged \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /lib/modules:/lib/modules:ro \
  ksense
```

Notes:
- The container needs `modprobe` (provided by `kmod`) and access to the host kernel headers/modules.
- If you see "Unable to find kernel headers", install headers on the host or enable `CONFIG_IKHEADERS=m`.

## Kubernetes

The DaemonSet runs one pod per node. Apply the namespace and DaemonSet with kustomize:

```bash
kubectl apply -k kubernetes/
```

`kernel_metrics.csv` is generated inside the pod filesystem when running as a DaemonSet.

Check status:

```bash
kubectl get ds -n ksense
kubectl get pods -n ksense -l app=ksense -o wide
```

## Research Context (IEEE Paper Summary)

K-Sense targets edge environments where applications are often black boxes and
application-level instrumentation is impractical. It relies on a small set of
kernel-level delay signals (CPU scheduling latency, SoftIRQ processing time, and
D-state I/O latency) and models the system state as a point in a multi-dimensional
feature space. The deviation from a calibrated baseline is quantified using the
Mahalanobis distance (friction), while a direction sign indicates whether the
system is above or below the baseline. An adaptive energy signal captures short-
term instability in friction.

The framework is evaluated with real workloads (DeathStarBench microservices and
a sentiment-analysis application). Results show that friction tracks application
P99 latency under changing load and remains informative when CPU utilization and
PSI saturate, enabling non-invasive QoS inference for admission control and
scheduling decisions.

## Notes

- Requires `bcc` and `numpy`.
- Run with appropriate privileges for eBPF (typically `sudo`).
