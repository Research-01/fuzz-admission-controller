#!/usr/bin/env python3
"""
Fuzzy Scheduler with Rate Limiting - FINAL PRODUCTION VERSION

FEATURES:
1. Rate limiting: Max N pods per cycle (default: 1)
2. Fresh score per pod: Re-queries for each placement
3. Stabilization delay: Wait M seconds between placements (default: 5s)
4. Crash prevention: Robust None checks and error handling
5. Clean logging: No SSL warnings, detailed status messages

CONFIGURATION (Environment Variables):
- FUZZY_SCHEDULER_NAME: Scheduler name (default: "fuzzy-scheduler")
- FUZZY_SCHEDULER_POLL_S: Seconds between cycles (default: 10)
- FUZZY_SCHEDULER_MAX_PER_CYCLE: Max pods to schedule per cycle (default: 1)
- FUZZY_SCHEDULER_PLACEMENT_DELAY_S: Seconds between placements (default: 5)
- FUZZY_WEBHOOK_SERVICE: Webhook service name (default: "ksense-fuzzy-webhook")
- FUZZY_WEBHOOK_NAMESPACE: Webhook namespace (default: "ksense")
- FUZZY_WEBHOOK_PORT: Webhook port (default: 8443)
- FUZZY_WEBHOOK_SCHEME: http or https (default: "https")
- FUZZY_INSECURE_TLS: Skip cert verification (default: "true")

VERSION: 3.0 Final
DATE: 2026-01-28
"""

import os
import sys
import time
import warnings

# Suppress SSL warnings (expected with self-signed certs)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from kubernetes import client, config

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# Configuration from environment variables
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
    """Load Kubernetes configuration (in-cluster or kubeconfig)"""
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()


def _list_webhook_endpoints(core):
    """
    List all webhook endpoint IPs from service endpoints.
    Returns: [(ip, port), ...]
    """
    try:
        ep = core.read_namespaced_endpoints(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
    except client.exceptions.ApiException as e:
        print(f"[fuzzy-scheduler] ✗ Failed to read endpoints: {e}")
        return []
    
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
    """
    Query a single webhook endpoint for fuzzy score.
    Returns: dict with {"score": float, "decision": str, ...} or None
    """
    url = f"{WEBHOOK_SCHEME}://{ip}:{port}/score"
    try:
        resp = requests.get(url, timeout=1.5, verify=not INSECURE)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        # Don't log every failure - webhook might be starting up
        return None


def _choose_node(core):
    """
    Choose best node based on current fuzzy scores.
    
    Returns: (node_name, score, report) tuple
    - (str, float, dict): Successful node selection
    - (None, None, None): No suitable node found (all denied or unreachable)
    """
    endpoints = _list_webhook_endpoints(core)
    
    if not endpoints:
        print(f"[fuzzy-scheduler] ✗ No webhook endpoints available")
        return None, None, None
    
    best = None
    all_reports = []
    
    # Query all endpoints
    for ip, port in endpoints:
        report = _score_endpoint(ip, port)
        if not report:
            continue
        all_reports.append((ip, port, report))
        
        # Skip nodes that deny
        if report.get("decision") == "deny":
            continue
        
        score = report.get("score")
        if score is None:
            continue
        
        # Track best (lowest) score
        if best is None or score < best["score"]:
            best = {"ip": ip, "port": port, "score": score, "report": report}
    
    # All nodes denied or unreachable
    if not best:
        if all_reports:
            # Log denial reasons
            reasons = []
            for ip, port, report in all_reports:
                score = report.get("score", "N/A")
                decision = report.get("decision", "unknown")
                reasons.append(f"{ip}:score={score},decision={decision}")
            print(f"[fuzzy-scheduler] All nodes denied: {'; '.join(reasons)}")
        return None, None, None
    
    # Map endpoint IP to node name
    try:
        ep = core.read_namespaced_endpoints(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
        for subset in ep.subsets or []:
            for addr in subset.addresses or []:
                if addr.ip == best["ip"] and addr.node_name:
                    return addr.node_name, best["score"], best["report"]
    except Exception as e:
        print(f"[fuzzy-scheduler] ✗ Failed to map IP to node: {e}")
    
    # Fallback: if only one node in cluster, use it
    try:
        nodes = core.list_node().items
        if len(nodes) == 1:
            return nodes[0].metadata.name, best["score"], best["report"]
    except Exception as e:
        print(f"[fuzzy-scheduler] ✗ Failed to list nodes: {e}")
    
    return None, None, None


def _pending_pods(core):
    """
    Get all pending pods that use this scheduler.
    Returns: list of V1Pod objects
    """
    try:
        pods = core.list_pod_for_all_namespaces().items
    except client.exceptions.ApiException as e:
        print(f"[fuzzy-scheduler] ✗ Failed to list pods: {e}")
        return []
    
    pending = []
    for pod in pods:
        # Only handle pods for this scheduler
        if pod.spec.scheduler_name != SCHEDULER_NAME:
            continue
        # Skip already scheduled pods
        if pod.spec.node_name:
            continue
        # Only Pending phase
        if pod.status.phase != "Pending":
            continue
        pending.append(pod)
    
    return pending


def _bind_pod(core, pod, node_name):
    """
    Bind a pod to a node.
    Raises: ApiException on failure
    """
    target = client.V1ObjectReference(kind="Node", api_version="v1", name=node_name)
    meta = client.V1ObjectMeta(name=pod.metadata.name)
    binding = client.V1Binding(target=target, metadata=meta)
    core.create_namespaced_binding(pod.metadata.namespace, binding)


def main():
    """Main scheduler loop"""
    _load_kube()
    core = client.CoreV1Api()
    
    # Print startup banner
    print("=" * 80)
    print(f"FUZZY SCHEDULER v3.0 - Production Ready")
    print("=" * 80)
    print(f"Scheduler name: {SCHEDULER_NAME}")
    print(f"Webhook: {WEBHOOK_SCHEME}://{WEBHOOK_SERVICE}.{WEBHOOK_NAMESPACE}:{WEBHOOK_PORT}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Max placements per cycle: {MAX_PLACEMENTS_PER_CYCLE}")
    print(f"Delay between placements: {DELAY_BETWEEN_PLACEMENTS_S}s")
    print(f"TLS verification: {'disabled' if INSECURE else 'enabled'}")
    print("=" * 80)
    print()
    
    while True:
        cycle_start = time.time()
        
        # Get pending pods
        pods = _pending_pods(core)
        
        if not pods:
            # No work to do
            time.sleep(POLL_INTERVAL)
            continue
        
        print(f"[fuzzy-scheduler] Found {len(pods)} pending pod(s)")
        
        placements_this_cycle = 0
        
        for pod in pods:
            # CRITICAL: Check rate limit BEFORE processing
            if placements_this_cycle >= MAX_PLACEMENTS_PER_CYCLE:
                remaining = len(pods) - pods.index(pod)
                print(f"[fuzzy-scheduler] Rate limit reached ({MAX_PLACEMENTS_PER_CYCLE} placements/cycle)")
                print(f"[fuzzy-scheduler] {remaining} pod(s) will be tried next cycle")
                break  # Stop processing, wait for next cycle
            
            # Query fresh score for THIS pod
            pod_key = f"{pod.metadata.namespace}/{pod.metadata.name}"
            print(f"[fuzzy-scheduler] Querying nodes for {pod_key}")
            node_name, score, report = _choose_node(core)
            
            # No suitable node available
            if not node_name:
                print(f"[fuzzy-scheduler] No suitable node for {pod.metadata.name} (all denied or unavailable)")
                continue  # Try next pod
            
            # Defensive check (should never happen, but be safe)
            if node_name is None:
                print(f"[fuzzy-scheduler] ✗ ERROR: node_name is None for {pod.metadata.name} - skipping")
                continue
            
            # Attempt to bind pod to node
            try:
                _bind_pod(core, pod, node_name)
                placements_this_cycle += 1  # Increment immediately after successful bind
                
                # Success logging
                print(f"[fuzzy-scheduler] ✓ Bound {pod_key} -> {node_name}")
                print(f"[fuzzy-scheduler]   Score: {score:.2f}, Level: {report.get('level', 'N/A')}")
                print(f"[fuzzy-scheduler]   Placements this cycle: {placements_this_cycle}/{MAX_PLACEMENTS_PER_CYCLE}")
                
                # Delay AFTER successful placement (let system stabilize)
                # Only delay if:
                # 1. We haven't reached the rate limit yet
                # 2. There are more pods to process
                if placements_this_cycle < MAX_PLACEMENTS_PER_CYCLE:
                    pods_remaining = len(pods) - pods.index(pod) - 1
                    if pods_remaining > 0:
                        print(f"[fuzzy-scheduler]   Waiting {DELAY_BETWEEN_PLACEMENTS_S}s for system to stabilize...")
                        time.sleep(DELAY_BETWEEN_PLACEMENTS_S)
                
            except client.exceptions.ApiException as exc:
                # Binding failed (pod might be gone, node might be unavailable, etc.)
                print(f"[fuzzy-scheduler] ✗ Bind failed for {pod.metadata.name}: {exc}")
                # Don't count as successful placement
                
            except Exception as exc:
                # Unexpected error
                print(f"[fuzzy-scheduler] ✗ Unexpected error for {pod.metadata.name}: {exc}")
                # Don't count as successful placement
        
        # Cycle summary
        cycle_duration = time.time() - cycle_start
        print(f"[fuzzy-scheduler] Cycle complete: {placements_this_cycle} placement(s) in {cycle_duration:.1f}s")
        
        # Sleep until next poll interval (account for time already spent)
        remaining_sleep = max(0, POLL_INTERVAL - cycle_duration)
        if remaining_sleep > 0:
            print(f"[fuzzy-scheduler] Sleeping {remaining_sleep:.1f}s until next cycle\n")
            time.sleep(remaining_sleep)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[fuzzy-scheduler] Shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"\n[fuzzy-scheduler] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
