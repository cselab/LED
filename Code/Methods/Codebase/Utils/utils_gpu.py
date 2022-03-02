#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import psutil
import os

# REPORTING GPU STATS
from threading import Thread
import time
import GPUtil


class GPUMonitor(Thread):
    def __init__(self, delay, multinode=False, rank_str=""):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.multinode = multinode
        self.rank_str = rank_str
        # The entire Python program exits when no alive non-daemon threads (master) are left.
        self.daemon = True
        self.start()

    def run(self):
        while not self.stopped:
            # GPUtil.showUtilization()
            output = ""
            GPUs = GPUtil.getGPUs()
            for i in range(len(GPUs)):
                GPU = GPUs[i]
                id_ = GPU.id
                name = GPU.name
                load = GPU.load
                memoryUtil = GPU.memoryUtil
                memoryUsed = GPU.memoryUsed
                memoryTotal = GPU.memoryTotal
                str_ = "[utils_gpu] GPU {:} | {:} | Load {:} | Memory {:.2f}% | {:}/{:} MB".format(
                    id_, name, load, memoryUtil, memoryUsed, memoryTotal)
                if i < len(GPUs) - 1: str_ += "\n"
                output += str_
            self.print(output)
            time.sleep(self.delay)

    def print(self, str_):
        if self.multinode:
            print("{:}{:}".format(self.rank_str, str_))
        else:
            print(str_)

    def stop(self):
        self.stopped = True


def getMemory():
    print("[utils_gpu] Memory tracking in MB...")
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024
    return memory
