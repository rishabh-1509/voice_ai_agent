import asyncio
from videosdk import Meeting, Events
from config import settings

class LiveConversationDemo:
    def __init__(self):
        self.meeting = Meeting(
            api_key=settings.VIDEOSDK_API_KEY,
            meeting_id=settings.VIDEOSDK_MEETING_ID
        )
        self.agent = ConversationAgent()

    async def start_demo(self):
        @self.meeting.on(Events.READY)
        async def on_ready():
            print("\n=== Conversation Started ===")
            await self._prompt_user()

        @self.meeting.on(Events.AUDIO_RECEIVED)
        async def on_audio(frame):
            text = await self._transcribe_audio(frame)
            response = await self.agent.process_input(text)
            await self._play_response(response)
            await self._prompt_user()

        await self.meeting.join()

    async def _prompt_user(self):
        print("\nUser: [Speaking...]")
        # Simulate voice input through terminal
        text = input("Type your message: ")
        await self._simulate_audio_input(text)

    async def _simulate_audio_input(self, text: str):
        # Convert text to mock audio frame
        mock_frame = text.encode()
        await self.meeting.send_audio(mock_frame)

    async def _transcribe_audio(self, frame: bytes) -> str:
        return frame.decode()  # Simplified for demo

    async def _play_response(self, text: str):
        print(f"\nAI: {text}")

async def main():
    demo = LiveConversationDemo()
    await demo.start_demo()

if __name__ == "__main__":
    asyncio.run(main())
