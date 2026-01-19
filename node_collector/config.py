AGGREGATION_WINDOW = 1
FLUSH_INTERVAL = 1
RUNQLEN_SAMPLE_NS = 200_000_000

LATENCY_THRESHOLDS = {
    "read": 100, "write": 100, "sendto": 50, "recvfrom": 50,
    "sendmsg": 50, "recvmsg": 50, "futex": 0, "epoll_wait": 0,
    "epoll_pwait": 0, "poll": 0, "ppoll": 0, "select": 0,
    "pselect6": 0, "nanosleep": 0, "fsync": 0, "fdatasync": 0,
    "connect": 50, "accept": 50, "accept4": 50, "stat": 0, "fstat": 0,
    "runqlen": 0, "softirq": 0, "sched_lat": 0, "block_io": 0,
    "tcp_retrans": 0, "dstate_io": 0, "io_qdepth": 0, "io_merge": 0,
}
