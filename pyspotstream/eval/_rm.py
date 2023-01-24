"""Monitor resource usage of a block."""
# SPDX-License-Identifier: AGPL-3.0-or-later

# import gc
import time
import tracemalloc

from typing import Optional

from dataclasses import dataclass


class ResourceMonitorError(Exception):
    pass


@dataclass
class ResourceUsage:
    name: Optional[str]  # Description of Usage
    time: float  # Measured in seconds
    memory: float  # Measured in bytes

    def __str__(self):
        if self.name is None:
            res = [f"Resource usage for {self.name}:"]
        else:
            res = ["Resource usage:"]
        res.append(f"  Time [s]: {self.time}")
        res.append(f"  Memory [b]: {self.memory}")
        return "\n".join(res)

    def __repr__(self):
        return str(self)


class ResourceMonitor:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.time = None
        self.memory = None
        self._start = None

    def __enter__(self):
        # FIXME: gc.collect is very expensive and in cursory testing had
        #        no effect on the measured memory usage. Leave it off for now.
        # gc.collect()
        if tracemalloc.is_tracing():
            raise ResourceMonitorError("Already tracing memory usage!")
        tracemalloc.start()
        tracemalloc.reset_peak()
        self._start = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.time = (time.perf_counter_ns() - self._start) / 1.0e9
        self.memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

    def result(self):
        if self.time is None or self.memory is None:
            raise ResourceMonitorError("No resources monitored yet.")
        return ResourceUsage(name=self.name, time=self.time, memory=self.memory)
