import GPUtil
import threading

class GPU:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.memoy_usage = []
        self.utilization = []
        self.monitoring = False
        self.thread = None

    def start_measure(self):
        self.monitoring = True
        self.clear_measurements()

        def measure_gpu():
            while self.monitoring:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.memoy_usage.append(gpus[0].memoryUsed)
                        self.utilization.append(gpus[0].load)
                except Exception as e:
                    print(f"Error while measuring GPU: {e}")
                    break
        
        self.thread = threading.Thread(target=measure_gpu)
        self.thread.start()

    def stop_measure(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def clear_measurements(self):
        self.memoy_usage = []
        self.utilization = []
    
    def get_memory_usage(self, peak: bool = False):
        if peak:
            return max(self.memoy_usage)
        return min(self.memoy_usage)
    
    def get_utilization(self, peak: bool = False):
        if peak:
            return max(self.utilization)
        return min(self.utilization)