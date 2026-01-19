# K-Sense Kernel Collector

This repo contains the K-Sense kernel metrics collector split into a small
Python package for easier maintenance.

## Layout

- `ksense/` Python package (BPF program, helpers, energy model, plotting, main loop)
- `main.py` Entry point script
- `graph/` Analysis outputs and plotting scripts

## Run

```bash
python3 main.py
```

## Notes

- Requires `bcc`, `numpy`, and `matplotlib` (for plotting).
- Run with appropriate privileges for eBPF (typically `sudo`).
