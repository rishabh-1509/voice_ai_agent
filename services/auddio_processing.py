import numpy as np
import webrtcvad
from collections import deque
from config import settings
from utilities.logger import logger

class AudioProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
        self.sample_rate = settings.SAMPLE_RATE
        self.frame_duration = settings.FRAME_DURATION
        self.buffers = {}
        
    def process_frame(self, participant_id: str, frame: bytes) -> bool:
        """Process audio frame with Voice Activity Detection"""
        if participant_id not in self.buffers:
            self.buffers[participant_id] = {
                'audio': deque(maxlen=50),
                'active': False
            }
            
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        buffer = self.buffers[participant_id]
        
        if is_speech:
            buffer['audio'].append(frame)
            if not buffer['active']:
                logger.info(f"Speech started for {participant_id}")
            buffer['active'] = True
            return True
        else:
            if buffer['active']:
                logger.info(f"Speech ended for {participant_id}")
            buffer['active'] = False
            return False

    def get_audio(self, participant_id: str) -> bytes:
        """Retrieve buffered audio for processing"""
        return b''.join(self.buffers[participant_id]['audio'])