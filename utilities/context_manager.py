from collections import defaultdict
from typing import Dict, List

class ContextManager:
    def __init__(self):
        self.history = defaultdict(list)
        self.entities = defaultdict(dict)

    def update(self, participant_id: str, text: str):
        self.history[participant_id].append(text)
        # Simple entity tracking
        if "meeting" in text.lower():
            self.entities[participant_id]['last_intent'] = 'booking'
        elif "profile" in text.lower():
            self.entities[participant_id]['last_intent'] = 'user_info'

    def get_context(self, participant_id: str) -> List[str]:
        return self.history[participant_id][-3:]