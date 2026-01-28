#!/usr/bin/env python3
"""
Improved Fuzzy Scheduler with Rate Limiting
 
FIXES:
1. Re-queries score after EACH pod placement
2. Configurable rate limit (max pods per interval)
3. Delay between placements to let system stabilize
4. Better logging for debugging
 
This prevents overwhelming the system by scheduling all pods at once.
"""
 
import os
import sys
import time
 
import requests
from kubernetes import client, config
 
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
 
 
SCHEDULER_NAME = os.getenv("FUZZY_SCHEDULER_NAME", "fuzzy-scheduler")
WEBHOOK_SERVICE = os.getenv("FUZZY_WEBHOOK_SERVICE", "ksense-fuzzy-webhook")
WEBHOOK_NAMESPACE = os.getenv("FUZZY_WEBHOOK_NAMESPACE", "ksense")
WEBHOOK_PORT = int(os.getenv("FUZZY_WEBHOOK_PORT", "8443"))
WEBHOOK_SCHEME = os.getenv("FUZZY_WEBHOOK_SCHEME", "https")
INSECURE = os.getenv("FUZZY_INSECURE_TLS", "true").lower() == "true"
POLL_INTERVAL = float(os.getenv("FUZZY_SCHEDULER_POLL_S", "10"))
 
# Rate limiting settings
MAX_PLACEMENTS_PER_CYCLE = int(os.getenv("FUZZY_SCHEDULER_MAX_PER_CYCLE", "1"))
DELAY_BETWEEN_PLACEMENTS_S = float(os.getenv("FUZZY_SCHEDULER_PLACEMENT_DELAY_S", "5"))
 
 
def _load_kube():
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()
 
 
def _list_webhook_endpoints(core):
    ep = core.read_namespaced_endpoints(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
    addrs = []
    for subset in ep.subsets or []:
        port = WEBHOOK_PORT
        if subset.ports:
            for p in subset.ports:
                if p.port:
                    port = p.port
                    break
        for addr in subset.addresses or []:
            addrs.append((addr.ip, port))
    return addrs
 
 
def _score_endpoint(ip, port):
    url = f"{WEBHOOK_SCHEME}://{ip}:{port}/score"
    try:
        resp = requests.get(url, timeout=1.5, verify=not INSECURE)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None
 
 
def _choose_node(core):
    """
    Choose best node based on current fuzzy scores.
    Returns: (node_name, score, report) or (None, None, None)
    """
    endpoints = _list_webhook_endpoints(core)
    best = None
    all_reports = []
    for ip, port in endpoints:
        report = _score_endpoint(ip, port)
        if not report:
            continue
        all_reports.append((ip, port, report))
        if report.get("decision") == "deny":
            continue
        score = report.get("score")
        if score is None:
            continue
        if best is None or score < best["score"]:
            best = {"ip": ip, "port": port, "score": score, "report": report}
    if not best:
        # Log why all nodes denied
        reasons = []
        for ip, port, report in all_reports:
            score = report.get("score", "N/A")
            decision = report.get("decision", "unknown")
            reasons.append(f"{ip}:score={score},decision={decision}")
        print(f"[fuzzy-scheduler] All nodes denied: {'; '.join(reasons)}")
        return None, None, None
    # Map endpoint IP to node
    ep = core.read_namespaced_endpoints(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
    for subset in ep.subsets or []:
        for addr in subset.addresses or []:
            if addr.ip == best["ip"] and addr.node_name:
                return addr.node_name, best["score"], best["report"]
    # Fallback
    nodes = core.list_node().items
    if len(nodes) == 1:
        return nodes[0].metadata.name, best["score"], best["report"]
    return None, None, None
 
 
def _pending_pods(core):
    pods = core.list_pod_for_all_namespaces().items
    pending = []
    for pod in pods:
        if pod.spec.scheduler_name != SCHEDULER_NAME:
            continue
        if pod.spec.node_name:
            continue
        if pod.status.phase != "Pending":
            continue
        pending.append(pod)
    return pending
 
 
def _bind_pod(core, pod, node_name):
    target = client.V1ObjectReference(kind="Node", api_version="v1", name=node_name)
    meta = client.V1ObjectMeta(name=pod.metadata.name)
    binding = client.V1Binding(target=target, metadata=meta)
    core.create_namespaced_binding(pod.metadata.namespace, binding)
 
 
def main():
    _load_kube()
    core = client.CoreV1Api()
    print("=" * 80)
    print(f"[fuzzy-scheduler] IMPROVED SCHEDULER with Rate Limiting")
    print("=" * 80)
    print(f"Scheduler name: {SCHEDULER_NAME}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Max placements per cycle: {MAX_PLACEMENTS_PER_CYCLE}")
    print(f"Delay between placements: {DELAY_BETWEEN_PLACEMENTS_S}s")
    print("=" * 80)
    print()
    while True:
        cycle_start = time.time()
        pods = _pending_pods(core)
        if not pods:
            print(f"[fuzzy-scheduler] No pending pods (sleeping {POLL_INTERVAL}s)")
            time.sleep(POLL_INTERVAL)
            continue
        print(f"[fuzzy-scheduler] Found {len(pods)} pending pod(s)")
        placements_this_cycle = 0
        for pod in pods:
            # Rate limit: stop after max placements
            if placements_this_cycle >= MAX_PLACEMENTS_PER_CYCLE:
                remaining = len(pods) - pods.index(pod)
                print(f"[fuzzy-scheduler] Rate limit reached ({MAX_PLACEMENTS_PER_CYCLE} placements/cycle)")
                print(f"[fuzzy-scheduler] {remaining} pod(s) will be tried next cycle")
                break
            # IMPORTANT: Re-query score for EACH pod
            print(f"[fuzzy-scheduler] Querying nodes for {pod.metadata.namespace}/{pod.metadata.name}")
            node_name, score, report = _choose_node(core)
            if not node_name:
                print(f"[fuzzy-scheduler] No suitable node for {pod.metadata.name} (all denied)")
                continue
            # Bind pod
            try:
                _bind_pod(core, pod, node_name)
                placements_this_cycle += 1
                print(f"[fuzzy-scheduler] ✓ Bound {pod.metadata.namespace}/{pod.metadata.name} -> {node_name}")
                print(f"[fuzzy-scheduler]   Score: {score:.2f}, Level: {report.get('level', 'N/A')}")
                # Delay before next placement (let system stabilize)
                if placements_this_cycle < MAX_PLACEMENTS_PER_CYCLE and pods.index(pod) < len(pods) - 1:
                    print(f"[fuzzy-scheduler]   Waiting {DELAY_BETWEEN_PLACEMENTS_S}s for system to stabilize...")
                    time.sleep(DELAY_BETWEEN_PLACEMENTS_S)
            except client.exceptions.ApiException as exc:
                print(f"[fuzzy-scheduler] ✗ Bind failed for {pod.metadata.name}: {exc}")
        # Summary
        cycle_duration = time.time() - cycle_start
        print(f"[fuzzy-scheduler] Cycle complete: {placements_this_cycle} placement(s) in {cycle_duration:.1f}s")
        # Wait until next poll interval
        remaining_sleep = max(0, POLL_INTERVAL - cycle_duration)
        if remaining_sleep > 0:
            print(f"[fuzzy-scheduler] Sleeping {remaining_sleep:.1f}s until next cycle\n")
            time.sleep(remaining_sleep)
 
 
if __name__ == "__main__":
    main()
