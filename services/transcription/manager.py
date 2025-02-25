import websockets
import json
import asyncio
from config import settings

class TranscriptionManager:
    def __init__(self):
        self.buffer = TranscriptionBuffer()
        self.connection = None
        
    async def connect(self):
        self.connection = await websockets.connect(
            f"wss://api.videosdk.live/v1/transcription/ws?"
            f"token={settings.videosdk_api_key}&"
            f"meetingId={settings.videosdk_meeting_id}"
        )
        await self._send_initial_config()
        return self.connection

    async def _send_initial_config(self):
        config_message = {
            "type": "configure_transcription",
            "languageCode": "en-US",
            "mode": "punctuated",
            "participantId": "ai-assistant"
        }
        await self.connection.send(json.dumps(config_message))

    async def process_messages(self, callback):
        try:
            async for message in self.connection:
                data = json.loads(message)
                if data.get("type") == "transcript" and data.get("isFinal"):
                    transcript = data.get("transcript", "").strip()
                    if transcript:
                        await callback(transcript)
        except websockets.exceptions.ConnectionClosed:
            print("Connection to transcription service closed")