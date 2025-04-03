from collections import deque
import psutil, threading, time
import numpy as np


class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.cpu_percentages = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.io_wait = deque(maxlen=100)
        self.cpu_per_core = deque(maxlen=100)  # Track per-core usage
        self.memory_bandwidth = deque(maxlen=100)
        self.io_stats = deque(maxlen=100)
        self.stop_flag = False
        self.process = psutil.Process()

    def start(self):
        self.stop_flag = False
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()

    def stop(self):
        self.stop_flag = True
        self.monitor_thread.join()

        # Calculate statistics with safety checks
        cpu_stats = {
            "mean": (
                np.mean(self.cpu_percentages)
                if len(self.cpu_percentages) > 0
                else 0
            ),
            "max": (
                np.max(self.cpu_percentages)
                if len(self.cpu_percentages) > 0
                else 0
            ),
            "min": (
                np.min(self.cpu_percentages)
                if len(self.cpu_percentages) > 0
                else 0
            ),
            "per_core_mean": (
                np.mean(self.cpu_per_core, axis=0)
                if len(self.cpu_per_core) > 0
                else np.zeros(psutil.cpu_count())
            ),
            "per_core_max": (
                np.max(self.cpu_per_core, axis=0)
                if len(self.cpu_per_core) > 0
                else np.zeros(psutil.cpu_count())
            ),
            "active_cores_mean": (
                np.mean(
                    [
                        sum(1 for x in cores if x > 10)
                        for cores in self.cpu_per_core
                    ]
                )
                if len(self.cpu_per_core) > 0
                else 0
            ),
        }

        mem_stats = {
            "mean": (
                np.mean(self.memory_usage) if len(self.memory_usage) > 0 else 0
            ),
            "max": (
                np.max(self.memory_usage) if len(self.memory_usage) > 0 else 0
            ),
            "min": (
                np.min(self.memory_usage) if len(self.memory_usage) > 0 else 0
            ),
        }

        io_stats = {
            "wait_mean": np.mean(self.io_wait) if len(self.io_wait) > 0 else 0,
            "wait_max": np.max(self.io_wait) if len(self.io_wait) > 0 else 0,
            "throughput_read": (
                np.mean(
                    [
                        sum(disk["read_bytes"] for disk in x["disk"].values())
                        / x["time_delta"]
                        for x in self.io_stats
                    ]
                )
                / 1024
                / 1024
                if len(self.io_stats) > 0
                else 0
            ),
            "throughput_write": (
                np.mean(
                    [
                        sum(disk["write_bytes"] for disk in x["disk"].values())
                        / x["time_delta"]
                        for x in self.io_stats
                    ]
                )
                / 1024
                / 1024
                if len(self.io_stats) > 0
                else 0
            ),
            "network_read": (
                np.mean(
                    [
                        x["network"]["bytes_recv"] / x["time_delta"]
                        for x in self.io_stats
                    ]
                )
                / 1024
                / 1024
                if len(self.io_stats) > 0
                else 0
            ),
            "network_write": (
                np.mean(
                    [
                        x["network"]["bytes_sent"] / x["time_delta"]
                        for x in self.io_stats
                    ]
                )
                / 1024
                / 1024
                if len(self.io_stats) > 0
                else 0
            ),
        }

        return {"cpu": cpu_stats, "memory": mem_stats, "io": io_stats}

    def _monitor(self):
        last_io = psutil.disk_io_counters(perdisk=True)  # Monitor per-disk
        last_net = psutil.net_io_counters()  # Add network monitoring
        last_time = time.time()

        while not self.stop_flag:
            # CPU usage (overall and per-core)
            per_core = psutil.cpu_percent(percpu=True)
            self.cpu_percentages.append(np.mean(per_core))
            self.cpu_per_core.append(per_core)

            # Memory usage
            mem = self.process.memory_info()
            self.memory_usage.append(mem.rss / 1024 / 1024)  # Convert to MB

            # IO wait and throughput
            cpu_times = psutil.cpu_times_percent()
            self.io_wait.append(cpu_times.iowait)

            # Enhanced IO monitoring
            current_io = psutil.disk_io_counters(perdisk=True)
            current_net = psutil.net_io_counters()
            current_time = time.time()

            # Calculate both disk and network throughput
            delta_time = current_time - last_time
            io_delta = {
                "disk": {
                    disk: {
                        "read_bytes": current_io[disk].read_bytes
                        - last_io[disk].read_bytes,
                        "write_bytes": current_io[disk].write_bytes
                        - last_io[disk].write_bytes,
                    }
                    for disk in current_io
                },
                "network": {
                    "bytes_recv": current_net.bytes_recv - last_net.bytes_recv,
                    "bytes_sent": current_net.bytes_sent - last_net.bytes_sent,
                },
                "time_delta": delta_time,
            }
            self.io_stats.append(io_delta)

            last_io = current_io
            last_net = current_net
            last_time = current_time

            time.sleep(self.interval)
