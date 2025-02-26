import asyncio
from collections import deque
import numpy as np
from videosdk import Meeting, Participant, Events
from config import settings
from utilities.logger import logger

class VideoSDKManager:
    def __init__(self):
        self.meeting = None
        self.participants = {}
        self.audio_buffers = {}
        self.processing_lock = asyncio.Lock()
        self.agent = None

    async def connect(self):
        self.meeting = Meeting(
            api_key=settings.VIDEOSDK_API_KEY,
            meeting_id=settings.VIDEOSDK_MEETING_ID
        )

        @self.meeting.on(Events.READY)
        async def on_ready():
            logger.info("Connected to meeting room")

        @self.meeting.on(Events.PARTICIPANT_JOINED)
        async def on_participant_joined(participant: Participant):
            self._register_participant(participant)

        await self.meeting.join()

    def _register_participant(self, participant: Participant):
        @participant.on(Events.AUDIO_RECEIVED)
        async def on_audio(frame):
            await self._process_audio_frame(participant.id, frame)
        
        self.participants[participant.id] = participant
        self.audio_buffers[participant.id] = {
            'buffer': deque(),
            'last_active': asyncio.get_event_loop().time()
        }
        logger.info(f"Participant joined: {participant.id}")

    async def _process_audio_frame(self, participant_id: str, frame: bytes):
        async with self.processing_lock:
            buffer_data = self.audio_buffers[participant_id]
            buffer_data['buffer'].append(frame)
            buffer_data['last_active'] = asyncio.get_event_loop().time()

            buffer_duration = len(buffer_data['buffer']) * settings.FRAME_DURATION / 1000
            time_since_last = asyncio.get_event_loop().time() - buffer_data['last_active']

            if buffer_duration >= settings.MIN_AUDIO_LENGTH or time_since_last > 0.5:
                await self._process_full_utterance(participant_id)

    async def _process_full_utterance(self, participant_id: str):
        buffer_data = self.audio_buffers[participant_id]
        audio_frames = list(buffer_data['buffer'])
        buffer_data['buffer'].clear()

        audio_data = np.frombuffer(b''.join(audio_frames), dtype=np.int16)
        transcript = await self._transcribe_audio(audio_data)
        
        if self.agent:
            response = await self.agent.process_input(participant_id, transcript)
            await self._play_audio_response(response)

    async def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        # Integrated with Whisper.cpp
        import whispercpp as w
        model = w.Whisper.from_pretrained("tiny.en")
        return model.transcribe(audio_data.flatten().tobytes())

    async def _play_audio_response(self, text: str):
        # Integrated TTS
        from gtts import gTTS
        import io
        with io.BytesIO() as audio_file:
            tts = gTTS(text=text, lang='en')
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            await self.meeting.play_audio(audio_file.read())

    async def close(self):
        await self.meeting.leave()