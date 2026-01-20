#!/usr/bin/env python3
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
POLL_INTERVAL = float(os.getenv("FUZZY_SCHEDULER_POLL_S", "2"))


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
    endpoints = _list_webhook_endpoints(core)
    best = None
    for ip, port in endpoints:
        report = _score_endpoint(ip, port)
        if not report:
            continue
        if report.get("decision") == "deny":
            continue
        score = report.get("score")
        if score is None:
            continue
        if best is None or score < best["score"]:
            best = {"ip": ip, "port": port, "score": score}
    if not best:
        return None
    # Map endpoint IP to node via EndpointAddress.node_name if available
    ep = core.read_namespaced_endpoints(WEBHOOK_SERVICE, WEBHOOK_NAMESPACE)
    for subset in ep.subsets or []:
        for addr in subset.addresses or []:
            if addr.ip == best["ip"] and addr.node_name:
                return addr.node_name
    # Fallback: if only one node, use it
    nodes = core.list_node().items
    if len(nodes) == 1:
        return nodes[0].metadata.name
    return None


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
    print(f"[fuzzy-scheduler] running as {SCHEDULER_NAME}")
    while True:
        pods = _pending_pods(core)
        for pod in pods:
            node_name = _choose_node(core)
            if not node_name:
                continue
            try:
                _bind_pod(core, pod, node_name)
                print(f"[fuzzy-scheduler] bound {pod.metadata.namespace}/{pod.metadata.name} -> {node_name}")
            except client.exceptions.ApiException as exc:
                print(f"[fuzzy-scheduler] bind failed: {exc}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
