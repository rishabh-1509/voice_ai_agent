import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import settings
from utilities.logger import logger

class HybridRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = self._init_vector_store()
        self.bm25_index = None
        self._setup_file_watcher()
        self._update_indexes()

    def _init_vector_store(self):
        try:
            return FAISS.load_local(
                settings.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            return FAISS.from_texts([""], self.embeddings)

    def _update_indexes(self):
        from langchain_community.document_loaders import DirectoryLoader

        loader = DirectoryLoader(settings.KNOWLEDGE_BASE_PATH, glob="**/*.txt")
        docs = loader.load()
        
        # Update vector store
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        
        # Update BM25
        texts = [doc.page_content for doc in docs]
        self.bm25_index = BM25Okapi([doc.split() for doc in texts])
        logger.info("Updated vector and keyword indexes")

    def _setup_file_watcher(self):
        class FileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(".txt"):
                    self._update_indexes()

        self.observer = Observer()
        self.observer.schedule(
            FileHandler(),
            path=settings.KNOWLEDGE_BASE_PATH,
            recursive=True
        )
        self.observer.start()

    def hybrid_search(self, query: str, top_k: int = 3) -> list:
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        bm25_scores = self.bm25_index.get_scores(query.split())
        bm25_results = [self.vector_store.docstore[i] for i in np.argsort(bm25_scores)[-top_k:]]
        return self._rerank(vector_results + vector_results)

    def _rerank(self, results):
        # Reciprocal Rank Fusion
        scores = {}
        for i, doc in enumerate(results):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1/(i+1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]