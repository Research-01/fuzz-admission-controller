#!/usr/bin/env python3
"""
random_lifetime_pods_with_services.py
 
IMPROVEMENTS:
1. CSV time-series logging of running pod counts
2. Configurable pod creation rate limiting
3. Staggered startup to avoid overwhelming scheduler
4. Better status tracking and reporting
 
Creates sentiment analysis pods (2-11) with corresponding services and random lifetimes.
Each pod gets its own NodePort service.
Pods are continuously cycled ONLY after they've been Running for their lifetime.
"""
 
import time
import random
import argparse
import csv
from datetime import datetime
from kubernetes import client, config
from kubernetes.client.rest import ApiException
 
 
NAMESPACE = "sa"
POD_NAME_PREFIX = "feedback-inference"
SERVICE_NAME_PREFIX = "feedback-inference"
IMAGE = "your-registry/feedback-inference:latest"
CONTAINER_PORT = 8000
SCHEDULER_NAME = "fuzzy-scheduler"
BASE_NODEPORT = 32000  # feedback-inference-2 gets 32002, etc.
 
 
def create_pod_manifest(pod_number: int) -> client.V1Pod:
    """Create a pod manifest for sentiment analysis"""
   
    pod_name = f"{POD_NAME_PREFIX}-{pod_number}"
   
    container = client.V1Container(
        name="feedback-inference",
        image=IMAGE,
        image_pull_policy="IfNotPresent",
        ports=[client.V1ContainerPort(container_port=CONTAINER_PORT)],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "128Mi"},
            limits={"cpu": "16", "memory": "5Gi"}  # ← REDUCED from 16cpu/10Gi to prevent memory issues
        )
    )
   
    spec = client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        scheduler_name=SCHEDULER_NAME
    )
   
    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(
            name=pod_name,
            namespace=NAMESPACE,
            labels={
                "app": pod_name,
                "qos": "be",
                "managed-by": "random-lifetime-script"
            }
        ),
        spec=spec
    )
   
    return pod
 
 
def create_service_manifest(pod_number: int) -> client.V1Service:
    """Create a service manifest for a pod"""
   
    service_name = f"{SERVICE_NAME_PREFIX}-{pod_number}-service"
    pod_name = f"{POD_NAME_PREFIX}-{pod_number}"
    node_port = BASE_NODEPORT + pod_number
   
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name=service_name,
            namespace=NAMESPACE,
            labels={
                "app": pod_name,
                "managed-by": "random-lifetime-script"
            }
        ),
        spec=client.V1ServiceSpec(
            type="NodePort",
            selector={"app": pod_name},
            ports=[
                client.V1ServicePort(
                    protocol="TCP",
                    port=80,
                    target_port=CONTAINER_PORT,
                    node_port=node_port
                )
            ]
        )
    )
   
    return service
 
 
def create_service(core_api: client.CoreV1Api, pod_number: int) -> bool:
    """Create a service, return True if successful"""
    service_name = f"{SERVICE_NAME_PREFIX}-{pod_number}-service"
    node_port = BASE_NODEPORT + pod_number
   
    try:
        service = create_service_manifest(pod_number)
        core_api.create_namespaced_service(namespace=NAMESPACE, body=service)
        print(f"[{timestamp()}] ✓ Created service: {service_name} (NodePort: {node_port})")
        return True
    except ApiException as e:
        if e.status == 409:
            print(f"[{timestamp()}] ⚠ Service {service_name} already exists")
            return True
        else:
            print(f"[{timestamp()}] ✗ Failed to create service {service_name}: {e}")
            return False
 
 
def delete_service(core_api: client.CoreV1Api, pod_number: int) -> bool:
    """Delete a service, return True if successful"""
    service_name = f"{SERVICE_NAME_PREFIX}-{pod_number}-service"
   
    try:
        core_api.delete_namespaced_service(name=service_name, namespace=NAMESPACE)
        print(f"[{timestamp()}] ✗ Deleted service: {service_name}")
        return True
    except ApiException as e:
        if e.status == 404:
            return True
        else:
            print(f"[{timestamp()}] ✗ Failed to delete service {service_name}: {e}")
            return False
 
 
def create_pod(core_api: client.CoreV1Api, pod_number: int) -> bool:
    """Create a pod, return True if successful"""
    pod_name = f"{POD_NAME_PREFIX}-{pod_number}"
   
    try:
        pod = create_pod_manifest(pod_number)
        core_api.create_namespaced_pod(namespace=NAMESPACE, body=pod)
        print(f"[{timestamp()}] ✓ Created pod: {pod_name}")
        return True
    except ApiException as e:
        if e.status == 409:
            print(f"[{timestamp()}] ⚠ Pod {pod_name} already exists, skipping")
            return False
        else:
            print(f"[{timestamp()}] ✗ Failed to create pod {pod_name}: {e}")
            return False
 
 
def delete_pod(core_api: client.CoreV1Api, pod_number: int) -> bool:
    """Delete a pod, return True if successful"""
    pod_name = f"{POD_NAME_PREFIX}-{pod_number}"
   
    try:
        core_api.delete_namespaced_pod(
            name=pod_name,
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(grace_period_seconds=5)
        )
        print(f"[{timestamp()}] ✗ Deleted pod: {pod_name}")
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        else:
            print(f"[{timestamp()}] ✗ Failed to delete pod {pod_name}: {e}")
            return False
 
 
def get_pod_info(core_api: client.CoreV1Api, pod_number: int) -> dict:
    """Get pod status and running time"""
    pod_name = f"{POD_NAME_PREFIX}-{pod_number}"
   
    try:
        pod = core_api.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)
        phase = pod.status.phase
       
        running_since = None
        if phase == "Running" and pod.status.start_time:
            running_since = pod.status.start_time.timestamp()
       
        age_seconds = None
        if pod.metadata.creation_timestamp:
            age_seconds = time.time() - pod.metadata.creation_timestamp.timestamp()
       
        return {
            'status': phase,
            'running_since': running_since,
            'age_seconds': age_seconds
        }
    except ApiException as e:
        if e.status == 404:
            return {'status': 'NotFound', 'running_since': None, 'age_seconds': None}
        return {'status': 'Unknown', 'running_since': None, 'age_seconds': None}
 
 
def count_pods_by_status(core_api: client.CoreV1Api, pod_numbers: list) -> dict:
    """Count pods by their status"""
    counts = {
        'Running': 0,
        'Pending': 0,
        'Failed': 0,
        'Succeeded': 0,
        'NotFound': 0,
        'Unknown': 0
    }
   
    for pod_num in pod_numbers:
        info = get_pod_info(core_api, pod_num)
        status = info['status']
        if status in counts:
            counts[status] += 1
   
    return counts
 
 
def write_csv_row(csv_file: str, pod_counts: dict):
    """Write a row to the CSV time-series file"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
    # Create file with header if it doesn't exist
    try:
        with open(csv_file, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time",
                "Running",
                "Pending",
                "Failed",
                "Succeeded",
                "NotFound",
                "Total_Managed"
            ])
    except FileExistsError:
        pass
   
    # Append data row
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            pod_counts['Running'],
            pod_counts['Pending'],
            pod_counts['Failed'],
            pod_counts['Succeeded'],
            pod_counts['NotFound'],
            sum(pod_counts.values())
        ])
 
 
def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Create SA pods with services and random lifetimes."
    )
    parser.add_argument("--min-lifetime", type=int, default=60,
                        help="Minimum pod RUNNING lifetime in seconds (default: 60)")
    parser.add_argument("--max-lifetime", type=int, default=180,
                        help="Maximum pod RUNNING lifetime in seconds (default: 180)")
    parser.add_argument("--startup-delay", type=int, default=10,
                        help="Delay between starting each initial pod (seconds, default: 10)")
    parser.add_argument("--recreate-delay", type=int, default=15,
                        help="Delay before recreating a pod after deletion (seconds, default: 15)")
    parser.add_argument("--csv-output", type=str, default="pod_lifecycle_timeseries.csv",
                        help="CSV file for time-series pod counts (default: pod_lifecycle_timeseries.csv)")
    parser.add_argument("--csv-interval", type=int, default=2,
                        help="Interval to write CSV rows (seconds, default: 5)")
    parser.add_argument("--max-concurrent-creations", type=int, default=2,
                        help="Maximum pods to create concurrently (default: 2)")
    parser.add_argument("--keep-services", action="store_true", default=True,
                        help="Keep services when stopping (default: True)")
    parser.add_argument("--delete-services", dest="keep_services", action="store_false",
                        help="Delete services when stopping")
   
    args = parser.parse_args()
   
    # Load Kubernetes config
    try:
        config.load_incluster_config()
        print("Loaded in-cluster config")
    except config.config_exception.ConfigException:
        config.load_kube_config()
        print("Loaded kubeconfig")
   
    core_api = client.CoreV1Api()
   
    # Pod numbers to manage: 2 through 11
    POD_NUMBERS = list(range(2, 12))
   
    print("=" * 80)
    print("Random Lifetime Pod Manager with Services (feedback-inference-2 to 11)")
    print("=" * 80)
    print(f"Namespace: {NAMESPACE}")
    print(f"Pods: feedback-inference-{POD_NUMBERS[0]} to feedback-inference-{POD_NUMBERS[-1]}")
    print(f"Services: NodePort {BASE_NODEPORT + POD_NUMBERS[0]} to {BASE_NODEPORT + POD_NUMBERS[-1]}")
    print(f"RUNNING lifetime range: {args.min_lifetime}-{args.max_lifetime} seconds")
    print(f"Startup delay: {args.startup_delay}s (prevents overwhelming scheduler)")
    print(f"Recreate delay: {args.recreate_delay}s (cooldown after deletion)")
    print(f"Max concurrent creations: {args.max_concurrent_creations}")
    print(f"CSV output: {args.csv_output} (updated every {args.csv_interval}s)")
    print(f"Scheduler: {SCHEDULER_NAME}")
    print(f"Resource limits: 2 CPU, 2Gi memory per pod")
    print("=" * 80)
    print()
   
    # Track pod lifecycles and creation cooldowns
    pod_tracker = {}
    creation_cooldown = {}  # {pod_number: timestamp_when_can_create}
   
    # Step 1: Create all services first
    print(f"[{timestamp()}] Creating services...")
    for pod_num in POD_NUMBERS:
        create_service(core_api, pod_num)
    print()
   
    # Step 2: Create initial pods with staggered startup
    print(f"[{timestamp()}] Creating initial pods (staggered)...")
    created_count = 0
    for pod_num in POD_NUMBERS:
        if create_pod(core_api, pod_num):
            lifetime = random.randint(args.min_lifetime, args.max_lifetime)
            pod_tracker[pod_num] = {
                "lifetime": lifetime,
                "running_since": None,
                "pending_since": time.time()
            }
            created_count += 1
            print(f"[{timestamp()}]   → feedback-inference-{pod_num} assigned lifetime: {lifetime}s")
           
            # Rate limit pod creation
            if created_count % args.max_concurrent_creations == 0:
                print(f"[{timestamp()}]   ⏸ Waiting {args.startup_delay}s before next batch...")
                time.sleep(args.startup_delay)
            else:
                time.sleep(2)  # Small delay between individual pods
   
    print(f"\n[{timestamp()}] Starting lifecycle management loop...\n")
   
    last_csv_write = time.time()
   
    try:
        while True:
            current_time = time.time()
           
            # Write CSV time-series data
            if current_time - last_csv_write >= args.csv_interval:
                pod_counts = count_pods_by_status(core_api, POD_NUMBERS)
                write_csv_row(args.csv_output, pod_counts)
                last_csv_write = current_time
           
            # Check each pod's status and lifetime
            for pod_num, info in list(pod_tracker.items()):
                # Check if pod is in creation cooldown
                if pod_num in creation_cooldown:
                    if current_time < creation_cooldown[pod_num]:
                        continue  # Skip this pod, still in cooldown
                    else:
                        del creation_cooldown[pod_num]  # Cooldown expired
               
                pod_info = get_pod_info(core_api, pod_num)
                status = pod_info['status']
               
                if status == "Running":
                    # Mark when pod started running
                    if info["running_since"] is None:
                        info["running_since"] = pod_info['running_since']
                        pending_duration = current_time - info["pending_since"]
                        print(f"[{timestamp()}] ✓ Pod feedback-inference-{pod_num} is now RUNNING "
                              f"(was pending for {pending_duration:.1f}s)")
                   
                    # Check if lifetime exceeded
                    running_duration = current_time - info["running_since"]
                    if running_duration >= info["lifetime"]:
                        print(f"[{timestamp()}] ⏱ Pod feedback-inference-{pod_num} reached lifetime "
                              f"({info['lifetime']}s, ran {running_duration:.1f}s)")
                       
                        # Delete pod
                        delete_pod(core_api, pod_num)
                       
                        # Set cooldown before recreation
                        creation_cooldown[pod_num] = current_time + args.recreate_delay
                        print(f"[{timestamp()}]   ⏸ Cooldown {args.recreate_delay}s before recreating...")
               
                elif status == "Pending":
                    pending_duration = current_time - info["pending_since"]
                    # Log every 30s to avoid spam
                    if int(pending_duration) % 30 == 0 and pending_duration >= 30:
                        print(f"[{timestamp()}] ⏳ Pod feedback-inference-{pod_num} still PENDING "
                              f"({pending_duration:.0f}s) - waiting for scheduler/resources")
               
                elif status in ["Failed", "Succeeded"]:
                    print(f"[{timestamp()}] ❌ Pod feedback-inference-{pod_num} is {status}, will recreate...")
                    delete_pod(core_api, pod_num)
                    creation_cooldown[pod_num] = current_time + args.recreate_delay
               
                elif status == "NotFound":
                    # Pod doesn't exist, create it (respecting cooldown)
                    print(f"[{timestamp()}] 🔄 Pod feedback-inference-{pod_num} not found, creating...")
                    new_lifetime = random.randint(args.min_lifetime, args.max_lifetime)
                    if create_pod(core_api, pod_num):
                        pod_tracker[pod_num] = {
                            "lifetime": new_lifetime,
                            "running_since": None,
                            "pending_since": current_time
                        }
                        print(f"[{timestamp()}]   → New lifetime: {new_lifetime}s")
           
            # Status summary every 30 seconds
            if int(current_time) % 30 == 0:
                pod_counts = count_pods_by_status(core_api, POD_NUMBERS)
                print(f"\n[{timestamp()}] === Status Summary ===")
                print(f"  Running: {pod_counts['Running']}, Pending: {pod_counts['Pending']}, "
                      f"Failed: {pod_counts['Failed']}, NotFound: {pod_counts['NotFound']}")
               
                for pod_num in sorted(pod_tracker.keys()):
                    info = pod_tracker[pod_num]
                    pod_info = get_pod_info(core_api, pod_num)
                    status = pod_info['status']
                   
                    if pod_num in creation_cooldown:
                        cooldown_left = int(creation_cooldown[pod_num] - current_time)
                        print(f"  feedback-inference-{pod_num}: COOLDOWN ({cooldown_left}s remaining)")
                    elif status == "Running" and info["running_since"]:
                        running_time = int(current_time - info["running_since"])
                        remaining = max(0, info["lifetime"] - running_time)
                        print(f"  feedback-inference-{pod_num}: {status:10s} | "
                              f"Ran: {running_time:3d}s | Remaining: {remaining:3d}s")
                    elif status == "Pending":
                        pending_time = int(current_time - info["pending_since"])
                        print(f"  feedback-inference-{pod_num}: {status:10s} | Pending: {pending_time:3d}s")
                    else:
                        print(f"  feedback-inference-{pod_num}: {status:10s}")
                print()
           
            time.sleep(1)
   
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] Stopping... Cleaning up pods...")
       
        # Write final CSV entry
        pod_counts = count_pods_by_status(core_api, POD_NUMBERS)
        write_csv_row(args.csv_output, pod_counts)
       
        for pod_num in pod_tracker.keys():
            delete_pod(core_api, pod_num)
       
        if not args.keep_services:
            print(f"[{timestamp()}] Deleting services...")
            for pod_num in POD_NUMBERS:
                delete_service(core_api, pod_num)
        else:
            print(f"[{timestamp()}] Keeping services (use --delete-services to remove)")
       
        print(f"[{timestamp()}] CSV saved to: {args.csv_output}")
        print(f"[{timestamp()}] Done.")
 
 
if __name__ == "__main__":
    main()