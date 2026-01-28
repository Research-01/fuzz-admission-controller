#!/usr/bin/env python3
"""
Fuzzy Scheduler with Rate Limiting - UPDATED (deterministic endpoint->node mapping)

Fixes included:
1) Deterministic mapping from webhook endpoint IP (pod IP) -> webhook pod -> pod.spec.nodeName
   using the Service selector (instead of Endpoints.addresses[].nodeName which is often empty).
2) Strict node_name normalization before binding (prevents `Invalid value for target`).
3) Light caching for Service selector + podIP->node map to reduce API calls.

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

Optional cache tuning:
- FUZZY_SELECTOR_CACHE_TTL_S (default: 15)
- FUZZY_PODMAP_CACHE_TTL_S (default: 5)

VERSION: 3.1
DATE: 2026-01-28
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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

# Cache tuning
_SELECTOR_CACHE_TTL_S = float(os.getenv("FUZZY_SELECTOR_CACHE_TTL_S", "15"))
_PODMAP_CACHE_TTL_S = float(os.getenv("FUZZY_PODMAP_CACHE_TTL_S", "5"))


def _load_kube():
    """Load Kubernetes configuration (in-cluster or kubeconfig)."""
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()


def _normalize_node_name(node_name) -> Optional[str]:
    """Return stripped node_name or None if invalid."""
    if node_name is None:
        return None
    s = str(node_name).strip()
    return s if s else None


def _list_webhook_endpoints(core: client.CoreV1Api) -> List[Tuple[str, int]]:
    """
    List all webhook endpoint IPs from Service Endpoints.
    Returns: [(ip, port), ...]
    """
    try:
        ep = core.read_namespaced_endpoints(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
    except client.exceptions.ApiException as e:
        print(f"[fuzzy-scheduler] ✗ Failed to read endpoints: {e}")
        return []

    addrs: List[Tuple[str, int]] = []
    for subset in ep.subsets or []:
        port = WEBHOOK_PORT
        if subset.ports:
            for p in subset.ports:
                if p.port:
                    port = p.port
                    break
        for addr in subset.addresses or []:
            if addr and addr.ip:
                addrs.append((addr.ip, port))
    return addrs


def _score_endpoint(ip: str, port: int) -> Optional[dict]:
    """
    Query a single webhook endpoint for fuzzy score.
    Returns: dict with {"score": float, "decision": str, ...} or None
    """
    url = f"{WEBHOOK_SCHEME}://{ip}:{port}/score"
    try:
        resp = requests.get(url, timeout=1.5, verify=not INSECURE)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        # Webhook might be starting up or temporarily unreachable
        return None


class WebhookPodResolver:
    """
    Resolve Service endpoint IPs (pod IPs) to node names:
        endpoint IP -> pod (by status.podIP) -> pod.spec.nodeName

    This is deterministic for selector-based Services.
    Uses small TTL caches to reduce API calls.
    """

    def __init__(self, core: client.CoreV1Api):
        self.core = core
        self._selector_cache: Tuple[float, Dict[str, str]] = (0.0, {})
        self._podmap_cache: Tuple[float, Dict[str, str]] = (0.0, {})

    def _get_service_selector(self) -> Dict[str, str]:
        now = time.time()
        cached_at, selector = self._selector_cache
        if selector and (now - cached_at) < _SELECTOR_CACHE_TTL_S:
            return selector

        try:
            svc = self.core.read_namespaced_service(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
            selector = svc.spec.selector or {}
        except Exception as e:
            print(f"[fuzzy-scheduler] ✗ Failed to read Service selector: {e}")
            selector = {}

        self._selector_cache = (now, selector)
        return selector

    def _get_pod_ip_to_node_map(self) -> Dict[str, str]:
        now = time.time()
        cached_at, podmap = self._podmap_cache
        if podmap and (now - cached_at) < _PODMAP_CACHE_TTL_S:
            return podmap

        selector = self._get_service_selector()
        if not selector:
            self._podmap_cache = (now, {})
            return {}

        label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])

        ip_to_node: Dict[str, str] = {}
        try:
            pods = self.core.list_namespaced_pod(WEBHOOK_NAMESPACE, label_selector=label_selector).items
            for p in pods:
                ip = getattr(p.status, "pod_ip", None)
                node = getattr(p.spec, "node_name", None)
                if ip and node:
                    ip_to_node[ip] = node
        except Exception as e:
            print(f"[fuzzy-scheduler] ✗ Failed to list webhook pods for selector '{label_selector}': {e}")
            ip_to_node = {}

        self._podmap_cache = (now, ip_to_node)
        return ip_to_node

    def endpoint_ip_to_node(self, endpoint_ip: str) -> Optional[str]:
        node = self._get_pod_ip_to_node_map().get(endpoint_ip)
        return _normalize_node_name(node)


def _choose_node(core: client.CoreV1Api, resolver: WebhookPodResolver):
    """
    Choose best node based on current fuzzy scores.

    Returns: (node_name, score, report) tuple
    - (str, float, dict): Successful node selection
    - (None, None, None): No suitable node found (all denied or unreachable)
    """
    endpoints = _list_webhook_endpoints(core)

    if not endpoints:
        print("[fuzzy-scheduler] ✗ No webhook endpoints available")
        return None, None, None

    best = None
    all_reports = []

    # Query all endpoints
    for ip, port in endpoints:
        report = _score_endpoint(ip, port)
        if not report:
            continue

        all_reports.append((ip, port, report))

        # Skip endpoints that deny
        if report.get("decision") == "deny":
            continue

        score = report.get("score")
        if score is None:
            continue

        # Track best (lowest) score
        if best is None or score < best["score"]:
            best = {"ip": ip, "port": port, "score": score, "report": report}

    # All denied/unreachable
    if not best:
        if all_reports:
            reasons = []
            for ip, _port, report in all_reports:
                score = report.get("score", "N/A")
                decision = report.get("decision", "unknown")
                reasons.append(f"{ip}:score={score},decision={decision}")
            print(f"[fuzzy-scheduler] All nodes denied: {'; '.join(reasons)}")
        return None, None, None

    # Deterministic mapping: endpoint IP (pod IP) -> webhook pod -> nodeName
    node_name = resolver.endpoint_ip_to_node(best["ip"])
    if node_name:
        return node_name, best["score"], best["report"]

    # If mapping fails, log useful info and return None (don’t bind with bad target)
    print(f"[fuzzy-scheduler] ✗ Could not map webhook endpoint IP {best['ip']} to a nodeName")
    return None, None, None


def _pending_pods(core: client.CoreV1Api):
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
        if pod.spec.scheduler_name != SCHEDULER_NAME:
            continue
        if pod.spec.node_name:
            continue
        if pod.status.phase != "Pending":
            continue
        pending.append(pod)

    return pending


def _bind_pod(core: client.CoreV1Api, pod, node_name: str):
    """
    Bind a pod to a node.
    Raises: ApiException on failure
    """
    node_name = _normalize_node_name(node_name)
    if not node_name:
        raise ValueError("node_name is empty/invalid after normalization")

    target = client.V1ObjectReference(kind="Node", api_version="v1", name=node_name)
    meta = client.V1ObjectMeta(name=pod.metadata.name)
    binding = client.V1Binding(target=target, metadata=meta)
    core.create_namespaced_binding(pod.metadata.namespace, binding)


def main():
    """Main scheduler loop."""
    _load_kube()
    core = client.CoreV1Api()
    resolver = WebhookPodResolver(core)

    # Startup banner
    print("=" * 80)
    print("FUZZY SCHEDULER v3.1 - Deterministic Mapping")
    print("=" * 80)
    print(f"Scheduler name: {SCHEDULER_NAME}")
    print(f"Webhook service: {WEBHOOK_SERVICE}.{WEBHOOK_NAMESPACE}:{WEBHOOK_PORT} ({WEBHOOK_SCHEME})")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Max placements per cycle: {MAX_PLACEMENTS_PER_CYCLE}")
    print(f"Delay between placements: {DELAY_BETWEEN_PLACEMENTS_S}s")
    print(f"TLS verification: {'disabled' if INSECURE else 'enabled'}")
    print("=" * 80)
    print()

    while True:
        cycle_start = time.time()

        pods = _pending_pods(core)
        if not pods:
            time.sleep(POLL_INTERVAL)
            continue

        print(f"[fuzzy-scheduler] Found {len(pods)} pending pod(s)")
        placements_this_cycle = 0

        for idx, pod in enumerate(pods):
            # Rate limit check
            if placements_this_cycle >= MAX_PLACEMENTS_PER_CYCLE:
                remaining = len(pods) - idx
                print(f"[fuzzy-scheduler] Rate limit reached ({MAX_PLACEMENTS_PER_CYCLE} placements/cycle)")
                print(f"[fuzzy-scheduler] {remaining} pod(s) will be tried next cycle")
                break

            pod_key = f"{pod.metadata.namespace}/{pod.metadata.name}"
            print(f"[fuzzy-scheduler] Querying nodes for {pod_key}")

            node_name, score, report = _choose_node(core, resolver)
            node_name = _normalize_node_name(node_name)

            if not node_name:
                print(f"[fuzzy-scheduler] No suitable node for {pod.metadata.name} (all denied/unavailable/unmappable)")
                continue

            try:
                _bind_pod(core, pod, node_name)
                placements_this_cycle += 1

                print(f"[fuzzy-scheduler] ✓ Bound {pod_key} -> {node_name}")
                try:
                    # score may be None if report is odd; be defensive
                    score_f = float(score) if score is not None else None
                    if score_f is not None:
                        print(f"[fuzzy-scheduler]   Score: {score_f:.2f}, Level: {report.get('level', 'N/A')}")
                    else:
                        print(f"[fuzzy-scheduler]   Score: N/A, Level: {report.get('level', 'N/A')}")
                except Exception:
                    print(f"[fuzzy-scheduler]   Score: {score}, Level: {report.get('level', 'N/A')}")

                print(f"[fuzzy-scheduler]   Placements this cycle: {placements_this_cycle}/{MAX_PLACEMENTS_PER_CYCLE}")

                # Stabilization delay if we might place another in same cycle
                if placements_this_cycle < MAX_PLACEMENTS_PER_CYCLE:
                    pods_remaining = len(pods) - idx - 1
                    if pods_remaining > 0:
                        print(f"[fuzzy-scheduler]   Waiting {DELAY_BETWEEN_PLACEMENTS_S}s for system to stabilize...")
                        time.sleep(DELAY_BETWEEN_PLACEMENTS_S)

            except client.exceptions.ApiException as exc:
                print(f"[fuzzy-scheduler] ✗ Bind failed for {pod.metadata.name}: {exc}")
            except Exception as exc:
                print(f"[fuzzy-scheduler] ✗ Unexpected error for {pod.metadata.name}: {exc}")

        cycle_duration = time.time() - cycle_start
        print(f"[fuzzy-scheduler] Cycle complete: {placements_this_cycle} placement(s) in {cycle_duration:.1f}s")

        remaining_sleep = max(0.0, POLL_INTERVAL - cycle_duration)
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
