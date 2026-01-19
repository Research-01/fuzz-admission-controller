# K-Sense Kernel Collector

This repo contains the K-Sense kernel metrics collector split into a small
Python package for easier maintenance.

## Layout

- `src/ksense/` Python package (BPF program, helpers, energy model, plotting, main loop)
- `main.py` Entry point script
- `scripts/monitor_cpu_psi.py` CPU + PSI monitoring helper script
- `scripts/step_response_prober.py` Step-response load generator for 3 apps (per-second CSV)
- `scripts/latency_p99_prober.py` Parallel P99 latency prober for 3 apps (per-interval CSV)
- `node_collector/` Node-level syscall + kernel signal collector (eBPF)
- `graph/` Analysis outputs and plotting scripts

## Run

```bash
python3 main.py
```

Node-level collector:

```bash
sudo -E python3 node_collector/main.py
```

Scripts:

```bash
python3 scripts/monitor_cpu_psi.py
python3 scripts/step_response_prober.py
python3 scripts/latency_p99_prober.py
```

## Notes

- Requires `bcc`, `numpy`, and `matplotlib` (for plotting).
- Run with appropriate privileges for eBPF (typically `sudo`).
