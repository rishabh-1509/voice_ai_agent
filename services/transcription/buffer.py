import time
from collections import deque

class TranscriptionBuffer:
    def __init__(self, window_size=5):
        self.buffer = []
        self.response_times = deque(maxlen=window_size)
        self.last_update_time = time.time()
        
        # Dynamic thresholds
        self.MIN_PAUSE = 2.0
        self.MAX_PAUSE = 10.0
        self.base_pause = 5.0

    def add(self, text):
        self.buffer.append(text)
        self._update_response_time()
        self.last_update_time = time.time()

    def _update_response_time(self):
        if len(self.buffer) > 1:
            new_response_time = time.time() - self.last_update_time
            self.response_times.append(new_response_time)
            
            # Calculate adaptive pause
            if self.response_times:
                avg_response = sum(self.response_times) / len(self.response_times)
                self.base_pause = min(
                    self.MAX_PAUSE,
                    max(self.MIN_PAUSE, avg_response * 0.75)
                )

    @property
    def current_threshold(self):
        return self.base_pause

    def time_since_last_update(self):
        return time.time() - self.last_update_time