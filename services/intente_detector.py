import re
from transformers import pipeline
from typing import Tuple
from config import settings
from utilities.logger import logger

class IntentDetector:
    def __init__(self):
        self.patterns = {
            'book_appointment': [
                r"(book|schedule)\s+(appointment|meeting)",
                r"set\s+up\s+a\s+call"
            ],
            'user_info': [
                r"my\s+(account|profile)",
                r"what\s+do\s+you\s+know\s+about\s+me"
            ]
        }
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def detect_intent(self, text: str) -> Tuple[str, float]:
        # Rule-based detection
        for intent, patterns in self.patterns.items():
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                return (intent, 1.0)
        
        # ML-based fallback
        result = self.classifier(
            text,
            candidate_labels=list(self.patterns.keys()) + ["general"],
            multi_label=False
        )
        return (result['labels'][0], result['scores'][0])