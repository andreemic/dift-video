
from prettytable import PrettyTable
import numpy as np
import time

class PerformanceManager():
    def __init__(self):
        self.time_arrays = {}
        self.temp_start_times = {}

    def start(self, name):
        self.temp_start_times[name] = time.time()

    def end(self, name):
        now = time.time()
        assert name in self.temp_start_times, f'start() must be called before end() for {name}'
        if name not in self.time_arrays:
            self.time_arrays[name] = []
        self.time_arrays[name].append(now - self.temp_start_times[name])
        del self.temp_start_times[name]

    def print_summary(self, title=None):
        table = PrettyTable()
        table.field_names = ["Name", "Average Time (s)", "95th Percentile Time (s)", "Num recorded", "Total time (s)"]
        if title is not None:
            table.title = title
        
        for name, times in self.time_arrays.items():
            avg_time = f'{np.mean(times):.4f}'
            percentile_95 = f'{np.percentile(times, 95):.4f}'
            num_recorded = len(times)
            total_time = f'{np.sum(times):.4f}'

            table.add_row([name, avg_time, percentile_95, num_recorded, total_time])

        print(table)
    def reset(self):
        self.time_arrays = {}
        self.temp_start_times = {}


class MockPerformanceManager():
    @staticmethod
    def start(name):
        pass
    
    @staticmethod
    def end(name):
        pass
    @staticmethod
    def print_summary():
        pass
