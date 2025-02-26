import asyncio
from services.video_sdk import VideoSDKManager
from agents.conversation_agent import ConversationAgent
from utilities.logger import logger

class AIAssistant:
    def __init__(self):
        self.video_sdk = VideoSDKManager()
        self.agent = ConversationAgent()
        self.video_sdk.agent = self.agent

    async def run(self):
        try:
            await self.video_sdk.connect()
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.shutdown()

    async def shutdown(self):
        await self.video_sdk.close()
        logger.info("System shutdown gracefully")

if __name__ == "__main__":
    assistant = AIAssistant()
    asyncio.run(assistant.run())