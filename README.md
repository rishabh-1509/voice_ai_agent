# Real-Time Conversational AI Agent with RAG & Intent-Based Function Execution

This repository contains a Python-based AI agent designed for real-time voice interactions. It integrates multiple cutting-edge technologies to provide a robust, context-aware conversational experience. Below is an overview of its key functionalities and components:

## Key Features

### Real-Time Conversational AI
- **Live Transcription via VideoSDK:**  
  Connects to the VideoSDK WebSocket to receive real-time transcription data, enabling live voice interactions.
- **Low-Latency Response Handling:**  
  Uses asynchronous programming (`asyncio`, `websockets`) to manage live input and deliver prompt responses.

### Interruption Handling
- **Seamless Conversation Management:**  
  Monitors conversation state and transcription buffers to gracefully handle user interruptions and partial inputs.
- **Dynamic Recovery:**  
  Clears outdated buffers and resumes processing without losing context when interruptions occur.

### Retrieval-Augmented Generation (RAG)
- **Enhanced Knowledge Retrieval:**  
  Utilizes a vector database (FAISS) and a HuggingFace LLM pipeline to retrieve and generate enriched responses based on external documents.
- **Context-Aware Responses:**  
  Employs a retrieval chain that leverages stored knowledge for answering general queries with improved accuracy.

### Intent-Based Function Execution
- **Intent Detection via Regex:**  
  Detects user intent (e.g., booking appointments, checking weather, retrieving user data, or telling a joke) using regex-based rules.
- **Dynamic API Integration:**  
  Executes corresponding functions based on detected intents, including:
  - **Booking & Canceling Appointments:**  
    Integrates with the Google Calendar API (with blocking calls offloaded to separate threads) for scheduling tasks.
  - **Weather Updates:**  
    Retrieves current weather conditions using the OpenWeather API via asynchronous HTTP calls.
  - **Joke Telling:**  
    Fetches random jokes from the Official Joke API.
  - **User Data Collection:**  
    Interactively collects and stores user data (e.g., email, phone, address) by prompting the user when data is missing.
  - **General Query Handling:**  
    Uses the RAG system to generate responses for queries that don’t match any specific intent.

### Voice Output
- **Text-to-Speech Conversion:**  
  Converts text responses to speech using gTTS, and plays the audio via the `playsound` module to deliver spoken feedback.

### Non-Blocking API Calls
- **Asynchronous HTTP Requests:**  
  Replaces blocking HTTP calls with `aiohttp` for non-blocking network I/O.
- **Thread Offloading:**  
  Wraps synchronous API calls (like Google Calendar operations) in `asyncio.to_thread` to avoid blocking the main event loop.
###File Strucure
ai-assistant/
├── main.py
├── config.py
├── requirements.txt
├── agents/
│   ├── __init__.py
│   └── conversation_agent.py
├── services/
│   ├── __init__.py
│   ├── video_sdk.py
│   ├── rag_engine.py
│   ├── intent_detector.py
│   └── audio_processor.py
├── integrations/
│   ├── __init__.py
│   ├── calendar.py
│   └── crm.py
└── utilities/
    ├── __init__.py
    ├── logger.py
    └── context_manager.py


This project demonstrates a scalable, multi-functional conversational AI agent capable of dynamically integrating various external APIs and knowledge sources. It's designed to be extended and adapted for diverse real-world applications.
