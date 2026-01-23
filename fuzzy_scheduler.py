#!/usr/bin/env python3
import os
import sys
import time
import traceback
from collections import Counter

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
K8S_TIMEOUT = float(os.getenv("FUZZY_K8S_TIMEOUT_S", "5.0"))
LOOP_LOG_EVERY = int(os.getenv("FUZZY_LOOP_LOG_EVERY", "1"))
SLOW_LOOP_S = float(os.getenv("FUZZY_SLOW_LOOP_S", "2.0"))
SLOW_CALL_S = float(os.getenv("FUZZY_SLOW_CALL_S", "0.5"))
STATS_EVERY = int(os.getenv("FUZZY_STATS_EVERY", "6"))

_STATS = Counter()


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print(f"[{_now()}] {msg}", flush=True)


def _timed(label, fn, *args, **kwargs):
    t0 = time.time()
    try:
        return fn(*args, **kwargs)
    finally:
        dt = time.time() - t0
        if dt > SLOW_CALL_S:
            _log(f"[fuzzy-scheduler] SLOW_CALL {label} elapsed={dt:.3f}s")


def _load_kube():
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()


def _list_webhook_endpoints(core):
    ep = _timed(
        "read_endpoints",
        core.read_namespaced_endpoints,
        WEBHOOK_SERVICE,
        WEBHOOK_NAMESPACE,
        _request_timeout=K8S_TIMEOUT,
    )
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
    if not addrs:
        _STATS["no_endpoints"] += 1
    return addrs


def _score_endpoint(ip, port):
    url = f"{WEBHOOK_SCHEME}://{ip}:{port}/score"
    try:
        resp = _timed(
            "score_endpoint",
            requests.get,
            url,
            timeout=1.5,
            verify=not INSECURE,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        _STATS["endpoint_unreachable"] += 1
        return None


def _node_from_endpoint_ip(core, ip):
    pods = _timed(
        "list_pods_by_ip",
        core.list_pod_for_all_namespaces,
        field_selector=f"status.podIP={ip}",
        _request_timeout=K8S_TIMEOUT,
    ).items
    if pods and pods[0].spec.node_name:
        return pods[0].spec.node_name
    return None


def _choose_node(core):
    endpoints = _list_webhook_endpoints(core)
    best = None
    for ip, port in endpoints:
        report = _score_endpoint(ip, port)
        if not report:
            continue
        if report.get("decision") == "deny":
            _STATS["decision_deny"] += 1
            continue
        score = report.get("score")
        if score is None:
            _STATS["score_missing"] += 1
            continue
        if best is None or score < best["score"]:
            best = {"ip": ip, "port": port, "score": score}
    if not best:
        _STATS["no_eligible_node"] += 1
        return None
    # Map endpoint IP to node via EndpointAddress.node_name if available
    ep = _timed(
        "read_endpoints_map",
        core.read_namespaced_endpoints,
        WEBHOOK_SERVICE,
        WEBHOOK_NAMESPACE,
        _request_timeout=K8S_TIMEOUT,
    )
    for subset in ep.subsets or []:
        for addr in subset.addresses or []:
            if addr.ip == best["ip"] and addr.node_name:
                return addr.node_name
    # Fallback: resolve endpoint IP -> pod -> node
    node_from_ip = _node_from_endpoint_ip(core, best["ip"])
    if node_from_ip:
        return node_from_ip
    # Fallback: if only one node, use it
    nodes = _timed("list_nodes", core.list_node, _request_timeout=K8S_TIMEOUT).items
    if len(nodes) == 1:
        return nodes[0].metadata.name
    _STATS["node_map_failed"] += 1
    return None


def _pending_pods(core):
    pods = _timed(
        "list_pods_pending",
        core.list_pod_for_all_namespaces,
        field_selector="status.phase=Pending",
        _request_timeout=K8S_TIMEOUT,
    ).items
    pending = []
    for pod in pods:
        if pod.metadata.deletion_timestamp:
            continue
        if pod.spec.scheduler_name != SCHEDULER_NAME:
            continue
        if pod.spec.node_name:
            continue
        pending.append(pod)
    return pending


def _bind_pod(core, pod, node_name):
    if not node_name:
        print(f"[fuzzy-scheduler] skip bind, empty node name for {pod.metadata.namespace}/{pod.metadata.name}")
        return False
    target = client.V1ObjectReference(kind="Node", api_version="v1", name=node_name)
    binding = client.V1Binding(
        metadata=client.V1ObjectMeta(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
        ),
        target=target,
    )
    if hasattr(core, "create_namespaced_pod_binding"):
        _timed(
            "create_pod_binding",
            core.create_namespaced_pod_binding,
            pod.metadata.name,
            pod.metadata.namespace,
            binding,
        )
    else:
        _timed(
            "create_binding",
            core.create_namespaced_binding,
            pod.metadata.namespace,
            binding,
        )
    return True


def main():
    _load_kube()
    core = client.CoreV1Api()
    _log(f"[fuzzy-scheduler] running as {SCHEDULER_NAME}")
    loop_i = 0
    while True:
        loop_i += 1
        loop_start = time.time()
        if loop_i % LOOP_LOG_EVERY == 0:
            _log(f"[fuzzy-scheduler] loop={loop_i} tick start poll_interval={POLL_INTERVAL}")
        try:
            pods = _pending_pods(core)
        except Exception as exc:
            _log(f"[fuzzy-scheduler] loop={loop_i} pending_pods ERROR: {type(exc).__name__}: {exc}")
            _log(traceback.format_exc())
            time.sleep(POLL_INTERVAL)
            continue

        _STATS["pods_seen"] += len(pods)
        _log(f"[fuzzy-scheduler] loop={loop_i} pending_pods={len(pods)}")

        for pod in pods:
            node_name = _choose_node(core)
            if not node_name:
                _log(f"[fuzzy-scheduler] no eligible node for {pod.metadata.namespace}/{pod.metadata.name}")
                continue
            try:
                if _bind_pod(core, pod, node_name):
                    _STATS["bound_ok"] += 1
                    _log(f"[fuzzy-scheduler] bound {pod.metadata.namespace}/{pod.metadata.name} -> {node_name}")
            except (client.exceptions.ApiException, ValueError) as exc:
                _STATS["bind_failed"] += 1
                _log(f"[fuzzy-scheduler] bind failed: {exc}")

        if loop_i % STATS_EVERY == 0:
            top = ", ".join([f"{k}={v}" for k, v in _STATS.most_common(10)])
            _log(f"[fuzzy-scheduler] stats {top}")

        elapsed = time.time() - loop_start
        if elapsed > SLOW_LOOP_S:
            _log(f"[fuzzy-scheduler] loop={loop_i} SLOW_LOOP elapsed={elapsed:.3f}s")
        if loop_i % LOOP_LOG_EVERY == 0:
            _log(f"[fuzzy-scheduler] loop={loop_i} tick end elapsed={elapsed:.3f}s")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
