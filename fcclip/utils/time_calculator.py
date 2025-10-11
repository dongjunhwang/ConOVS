import torch


class TimeCalculator(object):
    # ms = 1/1000 s
    def __init__(self):
        self.current_iter = 0
        self.mil_sec = 0.
        self.start_event = None
        self.end_event = None

    def time_start(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def time_end(self):
        self.end_event.record()
        self.end_event.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)
        self.save(elapsed_time)
        return elapsed_time

    def save(self, cur_mil_sec):
        if self.current_iter == 0:
            self.current_iter += 1
        else:
            self.mil_sec += cur_mil_sec
            self.current_iter += 1

    def return_avg_sec(self):
        return self.mil_sec / self.current_iter