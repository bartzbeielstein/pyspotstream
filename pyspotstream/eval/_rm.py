"""Monitor resource usage of a block."""

import gc
import tracemalloc
import time

from dataclasses import dataclass

@dataclass
class ResourceUsage:
    time: float   # Measured in seconds
    memory: float # Measured in bytes    
    
    def __str__(self):
        return f"Time used [s]: {self.time}\nMemory used [b]: {self.memory}"
    
    def __repr__(self):
        return str(self)   
    
class ResourceMonitor:
    def __init__(self):
        self.time = None
        self.memory = None
        self._start = None        
    
    def __enter__(self):
        if tracemalloc.is_tracing():
            raise Exception("Already tracing memory usage!")            
        tracemalloc.start()
        tracemalloc.reset_peak()
        self._start = time.time_ns()
        
    def __exit__(self, type, value, traceback):
        self.time = (time.time_ns() - self._start) / 1.0e9
        self.memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
    def result(self):
        if self.time is None or self.memory is None:
            raise Exception("No resources monitored yet.")
        return ResourceUsage(time=self.time, memory=self.memory)
