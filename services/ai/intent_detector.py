import re
from typing import Tuple, Dict, Any

class IntentDetector:
    def __init__(self):
        self.intent_patterns = self._load_intent_patterns()

    def _load_intent_patterns(self) -> Dict[str, Dict]:
        return {
            "book_appointment": {
                "patterns": [r"(?i)book\s+appointment", r"(?i)schedule\s+meeting"],
                "parameters": {
                    "date": r"\b(\d{4}-\d{2}-\d{2})\b",
                    "time": r"\b(\d{1,2}:\d{2})\b"
                }
            },
            "check_weather": {
                "patterns": [r"(?i)weather\s+in", r"(?i)forecast\s+for"],
                "parameters": {
                    "location": r"\b(in|at|for)\s+([\w\s]+)"
                }
            }
        }

    def detect_intent(self, text: str) -> Tuple[str, Dict[str, Any]]:
        text = text.lower()
        for intent, config in self.intent_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text):
                    params = self._extract_parameters(text, config["parameters"])
                    return intent, params
        return "general_query", {}

    def _extract_parameters(self, text: str, patterns: Dict[str, str]) -> Dict[str, str]:
        params = {}
        for param, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                params[param] = match.group(1)
        return params