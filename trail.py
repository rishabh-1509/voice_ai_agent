import asyncio
import websockets
import json
import time
import re
import datetime
import aiohttp  # For async HTTP requests
from typing import Dict, List, Any, Optional, Callable, Tuple
from gtts import gTTS
import os
import tempfile
from playsound import playsound
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# Import Google Calendar API client
from googleapiclient.discovery import build

# Configuration
class Config:
    # API settings
    VIDEOSDK_API_KEY = "70a16e9b-8d50-4f35-a82c-802cf2764fa6"
    VIDEOSDK_MEETING_ID = "h6c3-lpwr-5a9q"
    
    # LLM settings
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG settings
    DOCUMENTS_PATH = "./knowledge_base"
    INDEX_PATH = "./faiss_index"
    
    # Transcription settings
    SPEAKING_PAUSE_THRESHOLD = 5  # seconds
    CONVERSATION_END_THRESHOLD = 10  # seconds
    TRANSCRIPTION_LANGUAGE = "en-US"
    TRANSCRIPTION_MODE = "punctuated"
    PARTICIPANT_ID = "ai-assistant"
    
    # Intent detection settings
    INTENT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_INTENT = "general_query"
    
    # API endpoints and keys
    # For Google Calendar, we use the developer key (for demo purposes only)
    CALENDAR_API = "AIzaSyCXAlwDJdVkpqOwWKL7A33KOFiPoCzVt4Er"
    CRM_API = "https://api.example.com/crm"
    WEATHER_API = "https://api.openweathermap.org/data/2.5/weather"
    WEATHER_API_KEY = "5958f072c8e5c3364ed53d7971e19e27"

class TranscriptionBuffer:
    def __init__(self):
        self.buffer = []
        self.last_update_time = time.time()
    
    def add(self, text):
        self.buffer.append(text)
        self.last_update_time = time.time()
    
    def get_complete_transcription(self):
        return " ".join(self.buffer)
    
    def clear(self):
        self.buffer = []
        self.last_update_time = time.time()
    
    def time_since_last_update(self):
        return time.time() - self.last_update_time

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None
        self.setup_llm()
        self.setup_rag()
    
    def setup_llm(self):
        print("Loading LLM...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            token="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"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None
        ).to(device)
        
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            device=0 if device == "cuda" else -1
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        print("LLM loaded successfully")
    
    def setup_rag(self):
        if os.path.exists(self.config.INDEX_PATH):
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(self.config.INDEX_PATH, self.embeddings)
        else:
            print("Creating new vector store...")
            self._create_vector_store()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        print("RAG system initialized")
    
    def _create_vector_store(self):
        os.makedirs(self.config.DOCUMENTS_PATH, exist_ok=True)
        if not os.listdir(self.config.DOCUMENTS_PATH):
            with open(os.path.join(self.config.DOCUMENTS_PATH, "sample.txt"), "w") as f:
                f.write("This is a sample document for the RAG system.")
        
        loader = DirectoryLoader(self.config.DOCUMENTS_PATH, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.config.INDEX_PATH)
    
    def get_answer(self, query):
        try:
            result = self.qa_chain.invoke({"query": query})
            return result.get("result", "I don't have an answer for that.")
        except Exception as e:
            print(f"RAG Error: {e}")
            return "I'm having trouble retrieving information right now."

class IntentDetector:
    """Detects user intent from natural language inputs."""
    
    def __init__(self, config, llm):
        self.config = config
        self.llm = llm
        self.intent_patterns = self._load_intent_patterns()
        
    def _load_intent_patterns(self) -> Dict[str, Dict]:
        """Load regex patterns for basic intent matching."""
        return {
            # Appointment/scheduling intents
            "book_appointment": {
                "patterns": [
                    r"(?i)book\s+(?:an|a)?\s*appointment",
                    r"(?i)schedule\s+(?:a|an)?\s*meeting",
                    r"(?i)set\s+up\s+(?:a|an)?\s*(?:appointment|meeting)",
                    r"(?i)make\s+(?:a|an)?\s*(?:appointment|reservation)",
                ],
                "parameter_extractors": {
                    "date": r"(?i)(?:on|for)\s+(\d{4}-\d{2}-\d{2})",
                    "time": r"(?i)(?:at|by)\s+(\d{1,2}:\d{2})",
                    "duration": r"(?i)(?:for)\s+(\d+)\s+(?:hour|min(?:ute)?)",
                    "with_person": r"(?i)(?:with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                }
            },
            "cancel_appointment": {
                "patterns": [
                    r"(?i)cancel\s+(?:my|the)?\s*appointment",
                    r"(?i)reschedule\s+(?:my|the)?\s*(?:appointment|meeting)",
                ],
                "parameter_extractors": {
                    "date": r"(?i)(?:on|for)\s+(\d{4}-\d{2}-\d{2})",
                    "time": r"(?i)(?:at|by)\s+(\d{1,2}:\d{2})",
                    "with_person": r"(?i)(?:with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                    "event_id": r"(?i)(?:event\s*ID[:\-]?\s*)(\w+)"
                }
            },
            # Information retrieval intents
            "get_user_data": {
                "patterns": [
                    r"(?i)(?:show|get|find|display)\s+my\s+(?:information|data|profile)",
                    r"(?i)what\s+(?:information|data)\s+do\s+you\s+have\s+(?:about|on)\s+me",
                ],
                "parameter_extractors": {
                    "data_type": r"(?i)(?:specifically|just|only)\s+(?:my)?\s*(address|email|phone|contact|personal|medical|billing|payment)\s+(?:information|details|data)?",
                }
            },
            "check_weather": {
                "patterns": [
                    r"(?i)(?:what's|what\s+is|how's|how\s+is)\s+the\s+weather",
                    r"(?i)(?:will|is)\s+it\s+(?:rain|snow|sunny|cloudy)",
                    r"(?i)weather\s+(?:forecast|prediction|outlook)",
                ],
                "parameter_extractors": {
                    "location": r"(?i)(?:in|at|for|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s*,\s*[A-Z]{2})?)",
                    "date": r"(?i)(?:on|for)\s+(today|tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|this\s+weekend)",
                }
            },
            # Joke intent
            "tell_joke": {
                "patterns": [
                    r"(?i)tell\s+me\s+a\s+joke",
                    r"(?i)make\s+me\s+laugh",
                    r"(?i)joke"
                ],
                "parameter_extractors": {}
            },
            # General queries - fallback intent
            "general_query": {
                "patterns": [
                    r".*",  # Match anything as fallback
                ],
                "parameter_extractors": {}
            },
        }
    
    def detect_intent(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Detect the user's intent from their query.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Tuple of (intent_name, extracted_parameters, confidence_score)
        """
        # First try rule-based intent detection with regex patterns
        for intent_name, intent_data in self.intent_patterns.items():
            for pattern in intent_data["patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    # We found a matching intent, now extract parameters
                    params = {}
                    for param_name, param_pattern in intent_data["parameter_extractors"].items():
                        match = re.search(param_pattern, query)
                        if match:
                            params[param_name] = match.group(1)
                    
                    # Skip default fallback intent if we matched a specific intent
                    if intent_name != "general_query" or params:
                        return intent_name, params, 0.9  # High confidence for rule-based matches
        
        # If no specific intent matched, use the default intent
        return self.config.DEFAULT_INTENT, {}, 0.6

class FunctionExecutor:
    """Executes functions based on detected intent."""
    
    def __init__(self, config, rag_system):
        self.config = config
        self.rag_system = rag_system
        # Initialize a Google Calendar service using the developer key.
        try:
            self.calendar_service = build('calendar', 'v3', developerKey=self.config.CALENDAR_API)
        except Exception as e:
            print(f"Error initializing Google Calendar service: {e}")
            self.calendar_service = None
        self.function_map = self._initialize_function_map()
        # New: Store user profile information and track pending data field
        self.user_profile: Dict[str, str] = {}
        self.pending_data_field: Optional[str] = None
    
    def _initialize_function_map(self) -> Dict[str, Callable]:
        """Initialize the mapping between intents and functions."""
        return {
            "book_appointment": self.book_appointment,
            "cancel_appointment": self.cancel_appointment,
            "get_user_data": self.get_user_data,
            "check_weather": self.check_weather,
            "tell_joke": self.tell_joke,
            "general_query": self.handle_general_query
        }
    
    async def execute_function(self, intent: str, params: Dict[str, Any]) -> str:
        """
        Execute the appropriate function based on the detected intent.
        
        Args:
            intent: The detected intent
            params: Parameters extracted from the user query
            
        Returns:
            Response message to be spoken to the user
        """
        print(f"Executing function for intent: {intent} with params: {params}")
        
        if intent in self.function_map:
            try:
                return await self.function_map[intent](params)
            except Exception as e:
                print(f"Error executing function for intent {intent}: {e}")
                return f"I encountered an error while trying to {intent.replace('_', ' ')}. Please try again."
        else:
            return await self.handle_general_query(params)
    
    async def book_appointment(self, params: Dict[str, Any]) -> str:
        """Book an appointment using Google Calendar API."""
        try:
            date_str = params.get("date")
            time_str = params.get("time")
            duration = int(params.get("duration", 30))
            with_person = params.get("with_person", "someone")
            if not (date_str and time_str):
                return "Please provide both a date and time for the appointment."
            
            # Parse start datetime (expects date in YYYY-MM-DD and time in HH:MM 24-hour format)
            start_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            end_dt = start_dt + datetime.timedelta(minutes=duration)
            
            event = {
                'summary': f'Appointment with {with_person}',
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': 'UTC',
                },
            }
            if self.calendar_service:
                created_event = await asyncio.to_thread(
                    lambda: self.calendar_service.events().insert(calendarId='primary', body=event).execute()
                )
                event_id = created_event.get('id')
                return (f"I've booked your appointment with {with_person} on {date_str} at {time_str} "
                        f"for {duration} minutes. (Event ID: {event_id})")
            else:
                return "Calendar service is not available at the moment."
        except Exception as e:
            print(f"Error booking appointment: {e}")
            return "I encountered an issue while trying to book your appointment. Please try again later."
    
    async def cancel_appointment(self, params: Dict[str, Any]) -> str:
        """Cancel an appointment using Google Calendar API if an event ID is provided."""
        try:
            event_id = params.get("event_id")
            if event_id and self.calendar_service:
                await asyncio.to_thread(
                    lambda: self.calendar_service.events().delete(calendarId='primary', eventId=event_id).execute()
                )
                return f"I've canceled your appointment with event ID {event_id}."
            else:
                date = params.get("date", "")
                time_str = params.get("time", "")
                with_person = params.get("with_person", "")
                query_parts = []
                if date:
                    query_parts.append(f"on {date}")
                if time_str:
                    query_parts.append(f"at {time_str}")
                if with_person:
                    query_parts.append(f"with {with_person}")
                appointment_str = " ".join(query_parts)
                await asyncio.sleep(0.5)
                return f"I've canceled your appointment {appointment_str}."
        except Exception as e:
            print(f"Error canceling appointment: {e}")
            return "I couldn't cancel the appointment. Please try again later."
    
    async def get_user_data(self, params: Dict[str, Any]) -> str:
        """
        Retrieve user data based on request.
        If the requested data (e.g. email, phone, address) is not already stored,
        the agent asks the user to provide it and stores the answer.
        """
        data_type = params.get("data_type", "profile")
        # If data_type is "profile", you could return a summary.
        if data_type == "profile":
            if self.user_profile:
                summary = ", ".join(f"{k}: {v}" for k, v in self.user_profile.items())
                return f"Here is what I have on file: {summary}."
            else:
                return "I don't have any profile information on file yet."
        
        # For specific fields (e.g., email, phone, address)
        if data_type in self.user_profile:
            return f"Your {data_type} is {self.user_profile[data_type]}."
        else:
            # Mark this data field as pending so that the next user input will be stored.
            self.pending_data_field = data_type
            return f"I don't have your {data_type} on file. Could you please provide your {data_type}?"
    
    async def check_weather(self, params: Dict[str, Any]) -> str:
        """Check weather for specified location using the OpenWeather API."""
        location = params.get("location", "current location")
        try:
            url = (f"{self.config.WEATHER_API}?q={location}&appid={self.config.WEATHER_API_KEY}"
                   "&units=imperial")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        condition = data['weather'][0]['description']
                        temp = data['main']['temp']
                        return f"The weather in {location} is currently {condition} with a temperature of {temp}°F."
                    else:
                        return "I couldn't retrieve the weather information. Please try again later."
        except Exception as e:
            print(f"Error checking weather: {e}")
            return "I encountered an issue while checking the weather. Please try again later."
    
    async def tell_joke(self, params: Dict[str, Any]) -> str:
        """Tell a joke using the Official Joke API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://official-joke-api.appspot.com/random_joke") as response:
                    if response.status == 200:
                        joke_data = await response.json()
                        joke = f"{joke_data['setup']} {joke_data['punchline']}"
                        return joke
                    else:
                        return "I couldn't fetch a joke at the moment. Try again later."
        except Exception as e:
            print(f"Error fetching joke: {e}")
            return "I encountered an error while fetching a joke. Please try again later."
    
    async def handle_general_query(self, params: Dict[str, Any]) -> str:
        """Handle general queries using the RAG system."""
        query = params.get("query", "")
        if not query:
            return "I'm not sure what information you're looking for. Could you please rephrase your question?"
        return self.rag_system.get_answer(query)

async def async_speak(text):
    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)
    await loop.run_in_executor(None, playsound, fp.name)
    os.remove(fp.name)

async def process_transcription():
    config = Config()
    buffer = TranscriptionBuffer()
    rag = RAGSystem(config)
    intent_detector = IntentDetector(config, rag.llm)
    function_executor = FunctionExecutor(config, rag)
    
    ws_url = (f"wss://api.videosdk.live/v1/transcription/ws?"
              f"token={config.VIDEOSDK_API_KEY}&meetingId={config.VIDEOSDK_MEETING_ID}")
    
    print("Initializing AI assistant...")
    
    # Track conversation state to handle interruptions
    conversation_state = {
        "is_speaking": False,
        "current_query": "",
        "current_intent": "",
        "current_params": {},
        "last_response": "",
    }
    
    async with websockets.connect(ws_url) as ws:
        print("Connected to VideoSDK")
        await ws.send(json.dumps({
            "type": "configure_transcription",
            "languageCode": config.TRANSCRIPTION_LANGUAGE,
            "mode": config.TRANSCRIPTION_MODE,
            "participantId": config.PARTICIPANT_ID
        }))
        
        # Initial greeting
        welcome_message = "Hello! I'm your AI assistant. How can I help you today?"
        print(f"AI: {welcome_message}")
        await async_speak(welcome_message)
        
        while True:
            try:
                # Process the complete query when a pause is detected
                if buffer.time_since_last_update() > config.CONVERSATION_END_THRESHOLD and buffer.buffer:
                    query = buffer.get_complete_transcription()
                    conversation_state["current_query"] = query
                    print(f"Processing query: {query}")
                    
                    # Detect intent from the query
                    intent, params, confidence = intent_detector.detect_intent(query)
                    conversation_state["current_intent"] = intent
                    conversation_state["current_params"] = params
                    
                    # For general queries, pass the entire query
                    if intent == "general_query":
                        params["query"] = query
                    print(f"Detected intent: {intent} (confidence: {confidence:.2f})")
                    print(f"Extracted parameters: {params}")
                    
                    # Execute the appropriate function based on intent
                    response = await function_executor.execute_function(intent, params)
                    conversation_state["last_response"] = response
                    
                    print(f"Response: {response}")
                    conversation_state["is_speaking"] = True
                    await async_speak(response)
                    conversation_state["is_speaking"] = False
                    
                    buffer.clear()
                
                # Receive messages with timeout to regularly check conversation state
                message = await asyncio.wait_for(ws.recv(), timeout=1)
                data = json.loads(message)
                
                # Handle transcription data
                if data.get("type") == "transcript" and data.get("isFinal"):
                    transcript = data.get("transcript", "").strip()
                    if transcript:
                        print(f"User: {transcript}")
                        # If we're awaiting a user data response, store the answer
                        if function_executor.pending_data_field:
                            data_field = function_executor.pending_data_field
                            function_executor.user_profile[data_field] = transcript
                            function_executor.pending_data_field = None
                            confirmation = f"Thank you, I've saved your {data_field} as {transcript}."
                            print(f"AI: {confirmation}")
                            await async_speak(confirmation)
                            buffer.clear()
                            continue
                        
                        # Interrupt if AI is currently speaking
                        if conversation_state["is_speaking"]:
                            print("Detected interruption while speaking")
                            conversation_state["is_speaking"] = False
                            buffer.clear()
                        buffer.add(transcript)
                        
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(process_transcription())
