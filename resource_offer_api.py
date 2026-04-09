#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ksense.resource_offer_server import run_server


if __name__ == "__main__":
    host = os.getenv("RESOURCE_OFFER_HOST", "0.0.0.0")
    port = int(os.getenv("RESOURCE_OFFER_PORT", "8080"))
    run_server(host=host, port=port)
