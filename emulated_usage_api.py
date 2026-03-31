#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ksense.emulated_usage_api import run_server


if __name__ == "__main__":
    host = os.getenv("MZ_EMULATED_USAGE_HOST", "127.0.0.1")
    port = int(os.getenv("MZ_EMULATED_USAGE_PORT", "8090"))
    run_server(host=host, port=port)
