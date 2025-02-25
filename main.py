import asyncio
import json
import time
from config import settings
from services.transcription.manager import TranscriptionManager
from services.ai.rag_system import RAGSystem
from services.ai.intent_detector import IntentDetector
from services.ai.function_executor import FunctionExecutor
from services.tts import TTSService
from utilities.logger import setup_logger

class AIConversationManager:
    def __init__(self):
        # Initialize core components
        self.logger = setup_logger("AIConversationManager")
        self.transcription_manager = TranscriptionManager()
        self.rag = RAGSystem()
        self.intent_detector = IntentDetector()
        self.function_executor = FunctionExecutor()
        self.tts = TTSService()
        
        # Conversation state management
        self.conversation_state = {
            "is_speaking": False,
            "pending_action": None,
            "user_profile": {},
            "last_interaction": time.time(),
            "buffer": []
        }

    async def start(self):
        """Main entry point to start the AI assistant"""
        async with self.transcription_manager.connect() as ws:
            await self._send_greeting()
            await self.transcription_manager.process_messages(self.handle_transcript)

    async def _send_greeting(self):
        """Send initial greeting to the user"""
        greeting = "Hello! I'm your AI assistant. How can I help you today?"
        self.logger.info("AI: %s", greeting)
        await self.tts.speak(greeting)

    async def handle_transcript(self, transcript: str):
        """Process incoming transcription and generate response"""
        try:
            self.logger.info("User: %s", transcript)
            
            # Update conversation state
            self.conversation_state["last_interaction"] = time.time()
            self.conversation_state["buffer"].append(transcript)
            
            # Check for pending data collection
            if self.conversation_state["pending_action"]:
                await self._handle_pending_action(transcript)
                return
                
            # Detect intent and parameters
            intent, params = self.intent_detector.detect_intent(transcript)
            self.logger.debug("Detected intent: %s, params: %s", intent, params)
            
            # Execute appropriate action
            response = await self.function_executor.execute(intent, {
                **params,
                "user_profile": self.conversation_state["user_profile"],
                "raw_query": transcript
            })
            
            # Handle action results
            if isinstance(response, dict) and response.get("requires_followup"):
                self.conversation_state["pending_action"] = response
                prompt = response.get("prompt")
                await self.tts.speak(prompt)
            else:
                await self._send_response(response)
                
        except Exception as e:
            self.logger.error("Error processing transcript: %s", str(e))
            await self.tts.speak("Sorry, I encountered an error processing your request.")

    async def _handle_pending_action(self, transcript: str):
        """Handle follow-up responses from the user"""
        action = self.conversation_state["pending_action"]
        handler = getattr(self, f"_handle_{action['type']}", None)
        
        if handler:
            result = await handler(transcript, action)
            await self._send_response(result)
        else:
            await self.tts.speak("Sorry, I lost track of our conversation. Let's start over.")
            self.conversation_state["pending_action"] = None

    async def _handle_data_collection(self, transcript: str, action: dict):
        """Handle user data collection follow-up"""
        field = action["field"]
        self.conversation_state["user_profile"][field] = transcript
        self.conversation_state["pending_action"] = None
        return f"Thank you, I've updated your {field.replace('_', ' ')}."

    async def _send_response(self, response: str):
        """Send response to user with state management"""
        self.logger.info("AI: %s", response)
        self.conversation_state["is_speaking"] = True
        await self.tts.speak(response)
        self.conversation_state["is_speaking"] = False
        self.conversation_state["buffer"].clear()

if __name__ == "__main__":
    assistant = AIConversationManager()
    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\nAI assistant shutting down...")