from gtts import gTTS
import tempfile
import os
import asyncio
from playsound import playsound

class TTSService:
    async def speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            
        await asyncio.get_event_loop().run_in_executor(None, playsound, fp.name)
        os.remove(fp.name)