#!/usr/bin/env python3
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from node_collector.collector import main


if __name__ == "__main__":
    main()
