from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    # VideoSDK Configuration
    VIDEOSDK_API_KEY: str
    VIDEOSDK_MEETING_ID: str
    
    # AI Configuration
    LLM_MODEL: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Audio Processing
    SAMPLE_RATE: int = 16000
    FRAME_DURATION: int = 20  # ms
    MIN_AUDIO_LENGTH: float = 1.0  # seconds
    
    # Paths
    VECTOR_STORE_PATH: str = "./vector_store"
    KNOWLEDGE_BASE_PATH: str = "./knowledge_base"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()