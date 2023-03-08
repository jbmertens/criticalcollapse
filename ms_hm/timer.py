import time

class timer:
    """
    Just a simple class to time things.
    """

    def __init__(self):
        self.starts = {}
        self.times = {}
        self.starts["start"] = time.perf_counter()

    def start(self, timer_name) :
        self.starts[timer_name] = time.perf_counter()

    def stop(self, timer_name) :
        if timer_name not in self.times :
            self.times[timer_name] = 0
            
        self.times[timer_name] += time.perf_counter() - self.starts[timer_name]
        return self.times[timer_name]

    def results(self) :
        print("")
        print(" Timing results")
        print(" --------------")
        for name, t in self.times.items() :
            print("", name, "took", t)
