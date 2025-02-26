import pyaudio
import wave
import asyncio
from gtts import gTTS
import tempfile
import os

class TTSService:
    def __init__(self):
        self.py_audio = pyaudio.PyAudio()
        self.active_stream = None

    async def speak(self, text: str):
        """Convert text to interruptible speech"""
        try:
            # Generate speech file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                tts = gTTS(text=text, lang='en')
                tts.save(fp.name)
                
            # Stop any ongoing playback
            if self.active_stream:
                self.active_stream.stop_stream()
                self.active_stream.close()
                
            # Play audio with interrupt capability
            await self._play_audio_async(fp.name)
            os.remove(fp.name)
            
        except Exception as e:
            print(f"TTS Error: {e}")

    async def _play_audio_async(self, file_path: str):
        """Async wrapper for audio playback"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_audio, file_path)

    def _play_audio(self, file_path: str):
        """Blocking audio playback with PyAudio"""
        wf = wave.open(file_path, 'rb')
        
        self.active_stream = self.py_audio.open(
            format=self.py_audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        data = wf.readframes(1024)
        while data and self.active_stream.is_active():
            self.active_stream.write(data)
            data = wf.readframes(1024)

        self.active_stream.close()
        wf.close()