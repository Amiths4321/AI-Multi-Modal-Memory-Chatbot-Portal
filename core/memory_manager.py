import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # Official LangChain structure class
from config import Config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

# Inheriting from Embeddings makes this class natively compatible with LangChain FAISS
class LocalEmbeddingWrapper(Embeddings):
    def __init__(self, model_name):
        # Runs 100% locally on your machine's CPU
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        """Encodes an array of text strings into vectors"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text):
        """Encodes a single string query into a vector"""
        embedding = self.model.encode(text, show_progress_bar=False)
        return list(map(float, embedding))

class HybridMemoryManager:
    def __init__(self):
        self.embeddings = LocalEmbeddingWrapper(Config.EMBEDDING_MODEL)
        self.index_path = Config.FAISS_INDEX_PATH
        self.vector_store = self._load_or_create_store()

    def _load_or_create_store(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        expected_file = os.path.join(self.index_path, "index.faiss")
        
        # We now pass the entire robust object wrapper safely
        if os.path.exists(expected_file):
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
                
            initial_doc = [Document(page_content="Local System Initialized", metadata={"type": "system"})]
            db = FAISS.from_documents(initial_doc, self.embeddings) 
            db.save_local(self.index_path)
            return db

    def save_memory(self, user_input, model_response):
        """Saves memory chunks strictly to your local data folder"""
        memory_text = f"User context: {user_input} | Assistant response: {model_response}"
        doc = Document(page_content=memory_text, metadata={"timestamp": "local_session"})
        
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.index_path)

    def retrieve_relevant_memories(self, query, top_k=3):
        """Queries the local FAISS index file on your machine"""
        if not self.vector_store:
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
            return [d.page_content for d in docs if d.metadata.get("type") != "system"]
        except Exception:
            return []