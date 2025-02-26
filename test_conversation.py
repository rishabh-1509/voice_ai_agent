import asyncio
from config import settings
from services.video_sdk import VideoSDKManager
from agents.conversation_agent import ConversationAgent
from utilities.logger import setup_logger

logger = setup_logger("TestScript")

class TestRunner:
    def __init__(self):
        self.video_sdk = VideoSDKManager()
        self.agent = ConversationAgent()
        self.test_cases = [
            ("Can you book a meeting tomorrow at 2PM?", "booking"),
            ("What's my current account balance?", "crm"),
            ("Explain our refund policy", "rag")
        ]

    async def simulate_conversation(self):
        """Simulates end-to-end conversation flow"""
        try:
            # Connect to test meeting
            await self.video_sdk.connect()
            logger.info("Connected to test meeting")

            # Simulate participant joining
            test_participant = {
                "id": "user_123",
                "name": "Test User"
            }
            self.video_sdk._register_participant(test_participant)

            for query, expected_type in self.test_cases:
                # Simulate audio input
                logger.info(f"\nUser: {query}")
                
                # Process through agent
                response = await self.agent.process_input(query)
                
                # Verify response type
                response_type = self._classify_response(response)
                assert response_type == expected_type, \
                    f"Expected {expected_type} got {response_type}"
                
                logger.info(f"AI: {response}")
                logger.info(f"Passed: {query[:20]}... => {response_type}")

            logger.info("All test cases passed!")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
        finally:
            await self.video_sdk.meeting.leave()

    def _classify_response(self, response: str) -> str:
        """Classifies response type for validation"""
        if "booked" in response.lower():
            return "booking"
        if "balance" in response.lower():
            return "crm"
        return "rag"

if __name__ == "__main__":
    tester = TestRunner()
    asyncio.run(tester.simulate_conversation())
