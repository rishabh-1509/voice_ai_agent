from pydantic import BaseSettings

class Settings(BaseSettings):
    # VideoSDK Configuration
    videosdk_api_key: str = "70a16e9b-8d50-4f35-a82c-802cf2764fa6"
    videosdk_meeting_id: str = "h6c3-lpwr-5a9q"
    
    # AI Model Configuration
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Configuration
    documents_path: str = "./knowledge_base"
    index_path: str = "./faiss_index"
    
    # Conversation Settings
    speaking_pause_threshold: int = 5
    conversation_end_threshold: int = 10
    
    # API Keys
    calendar_api_key: str = "AIzaSyCXAlwDJdVkpqOwWKL7A33KOFiPoCzVt4Er"
    weather_api_key: str = "5958f072c8e5c3364ed53d7971e19e27"
    weather_api_base: str = "https://api.openweathermap.org/data/2.5/weather"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "ai_assistant.log"
    
    # Authentication Tokens
    hf_token: str = ""  # Should be set via environment variable
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()