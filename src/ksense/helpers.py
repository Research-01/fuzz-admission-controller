import csv
import os
from typing import Iterable



def ensure_csv(path: str, header: Iterable[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def percentiles_from_subbucket_hist(items, ps=(0.95, 0.99), subbits=4, mode="mid"):
    """
    Percentiles from histogram keyed by (b<<subbits)|s, where
      b = log2 bucket
      s = sub-bucket within that power-of-two range

    mode:
      - "lower": return lower edge of sub-bucket
      - "mid":   return mid-point of sub-bucket (recommended)
      - "upper": return upper edge of sub-bucket
    """
    buckets = sorted((int(k.value), int(v.value)) for k, v in items)
    total = sum(c for _, c in buckets)
    if total == 0:
        return {p: 0 for p in ps}

    targets = {p: int(total * p + 0.999999) for p in ps}
    out = {}
    running = 0

    subbuckets = 1 << int(subbits)

    def decode_range(key: int):
        b = key >> subbits
        s = key & (subbuckets - 1)

        # Base range for bucket b: [2^b, 2^(b+1)-1]
        lo = 1 << b
        hi = (1 << (b + 1)) - 1
        width = hi - lo + 1

        # Sub-range [sub_lo, sub_hi]
        sub_lo = lo + (width * s) // subbuckets
        sub_hi = lo + (width * (s + 1)) // subbuckets - 1
        if sub_hi < sub_lo:
            sub_hi = sub_lo

        return sub_lo, sub_hi

    def pick_value(key: int) -> int:
        sub_lo, sub_hi = decode_range(key)
        if mode == "lower":
            return sub_lo
        if mode == "upper":
            return sub_hi
        # mid
        return (sub_lo + sub_hi) // 2

    for key, c in buckets:
        running += c
        for p, t in targets.items():
            if p not in out and running >= t:
                out[p] = pick_value(key)

    # Ensure all percentiles present
    last_key = buckets[-1][0]
    for p in ps:
        out.setdefault(p, pick_value(last_key))
    return out
