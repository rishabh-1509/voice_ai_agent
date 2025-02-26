import os
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import settings
from utilities.logger import setup_logger

class DocumentUpdateHandler(FileSystemEventHandler):
    """Handles document directory changes for hot-reloading"""
    def __init__(self, reload_callback):
        super().__init__()
        self.reload_callback = reload_callback
        self.logger = setup_logger("DocumentWatcher")

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".txt"):
            self.logger.info(f"Detected change in {event.src_path}")
            self.reload_callback()

class RAGSystem:
    def __init__(self):
        self.logger = setup_logger("RAGSystem")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.vectorstore = None
        self.qa_chain = None
        self.observer = None
        
        # Initialize components
        self._setup_llm()
        self._setup_vector_store()
        self._setup_file_watcher()
        self.logger.info("RAG system initialized")

    def _setup_llm(self):
        """Initialize the language model pipeline"""
        self.logger.info("Loading LLM...")
        
        # Model loading
        tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model,
            token=os.getenv("HF_TOKEN", "")
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )

        # Pipeline configuration
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
        self.logger.info(f"LLM loaded on {device.upper()}")

    def _setup_vector_store(self):
        """Initialize or load vector store"""
        if os.path.exists(settings.index_path):
            self.logger.info("Loading existing vector store")
            self.vectorstore = FAISS.load_local(
                settings.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.logger.info("Creating new vector store")
            self.vectorstore = self._create_vector_store()
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.25}
            ),
            return_source_documents=False
        )

    def _create_vector_store(self):
        """Create vector store from documents"""
        self.logger.info(f"Loading documents from {settings.documents_path}")
        
        # Ensure documents directory exists
        os.makedirs(settings.documents_path, exist_ok=True)
        if not os.listdir(settings.documents_path):
            self._create_sample_document()

        # Load and split documents
        loader = DirectoryLoader(
            settings.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)

        # Create and save vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(settings.index_path)
        return vectorstore

    def _setup_file_watcher(self):
        """Initialize file system watcher for auto-reloading"""
        self.logger.info("Starting document watcher")
        self.observer = Observer()
        event_handler = DocumentUpdateHandler(self.reload_vector_store)
        self.observer.schedule(
            event_handler,
            path=settings.documents_path,
            recursive=True
        )
        self.observer.start()

    def reload_vector_store(self):
        """Hot-reload vector store when documents change"""
        self.logger.info("Reloading vector store...")
        try:
            self.vectorstore = self._create_vector_store()
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5, "lambda_mult": 0.25}
                )
            )
            self.logger.info("Vector store reloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to reload vector store: {str(e)}")

    def _create_sample_document(self):
        """Create sample document if directory is empty"""
        sample_path = os.path.join(settings.documents_path, "sample.txt")
        with open(sample_path, "w") as f:
            f.write("This is a sample document for the RAG system.\n")
            f.write("It demonstrates how knowledge base documents should be formatted.\n")
        self.logger.info("Created sample document")

    def get_answer(self, query: str) -> str:
        """Get answer from RAG system"""
        try:
            result = self.qa_chain.invoke({"query": query})
            return result.get("result", "I don't have an answer for that.")
        except Exception as e:
            self.logger.error(f"RAG Error: {str(e)}")
            return "I'm having trouble retrieving information right now."

    def __del__(self):
        """Cleanup resources"""
        if self.observer:
            self.observer.stop()
            self.observer.join()