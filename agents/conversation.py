from services import HybridRAG, IntentDetector
from integrations import CalendarManager, CRMClient
from utilities.context_manager import ContextManager
from config import settings
from utilities.logger import logger

class ConversationAgent:
    def __init__(self):
        self.rag = HybridRAG()
        self.intent_detector = IntentDetector()
        self.calendar = CalendarManager()
        self.crm = CRMClient()
        self.context = ContextManager()

    async def process_input(self, participant_id: str, text: str) -> str:
        self.context.update(participant_id, text)
        
        intent, confidence = self.intent_detector.detect_intent(text)
        
        if confidence < 0.7:
            return await self._handle_general_query(text)
            
        match intent:
            case 'book_appointment':
                return await self._handle_booking(text)
            case 'user_info':
                return self._handle_user_info(participant_id)
            case _:
                return await self._handle_general_query(text)

    async def _handle_booking(self, text: str) -> str:
        params = self._extract_parameters(text)
        event_id = await self.calendar.create_event(params)
        return f"Meeting booked! ID: {event_id}"

    def _handle_user_info(self, participant_id: str) -> str:
        profile = self.crm.get_profile(participant_id)
        return f"Profile: {profile}"

    async def _handle_general_query(self, text: str) -> str:
        results = self.rag.hybrid_search(text)
        return "\n".join([f"- {res[0]}" for res in results[:3]])

    def _extract_parameters(self, text: str) -> dict:
        # Implement entity extraction
        return {"title": "Meeting", "time": "now"}