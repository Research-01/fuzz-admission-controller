# K-Sense Kernel Collector

This repo contains the K-Sense kernel metrics collector split into a small
Python package for easier maintenance.

## Layout

- `src/ksense/` Python package (BPF program, helpers, energy model, plotting, main loop)
- `main.py` Entry point script
- `scripts/monitor_cpu_psi.py` CPU + PSI monitoring helper script
- `scripts/step_response_prober.py` Step-response load generator for 3 apps (per-second CSV)
- `scripts/latency_p99_prober.py` Parallel P99 latency prober for 3 apps (per-interval CSV)
- `graph/` Analysis outputs and plotting scripts

## Run

```bash
python3 main.py
```

## Notes

- Requires `bcc`, `numpy`, and `matplotlib` (for plotting).
- Run with appropriate privileges for eBPF (typically `sudo`).
