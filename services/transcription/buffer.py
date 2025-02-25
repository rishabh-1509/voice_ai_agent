import time

class TranscriptionBuffer:
    def __init__(self):
        self.buffer = []
        self.last_update_time = time.time()
    
    def add(self, text):
        self.buffer.append(text)
        self.last_update_time = time.time()
    
    def get_complete_transcription(self):
        return " ".join(self.buffer)
    
    def clear(self):
        self.buffer = []
        self.last_update_time = time.time()
    
    def time_since_last_update(self):
        return time.time() - self.last_update_time