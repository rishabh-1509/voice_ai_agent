from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from config import settings

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.vectorstore = self._setup_vector_store()
        self.llm = self._setup_llm()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        
    def _setup_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model,
            token=os.getenv("HF_TOKEN")
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7
        )
        
    def _setup_vector_store(self):
        if os.path.exists(settings.index_path):
            return FAISS.load_local(settings.index_path, self.embeddings)
        else:
            from langchain_community.document_loaders import DirectoryLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            loader = DirectoryLoader(settings.documents_path, glob="**/*.txt")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(settings.index_path)
            return vectorstore
            
    def get_answer(self, query):
        return self.qa_chain.run(query)