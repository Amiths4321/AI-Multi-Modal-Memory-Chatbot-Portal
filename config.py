class Config:
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    
    # --- REMOTE GPU SERVER CONFIGURATION ---
    # Points to your remote deployment (vLLM, Ollama, or custom FastAPI wrapper running Qwen-VL)
    REMOTE_SERVER_URL = "http://10.22.39.192:11434"  # Update port if using vLLM (typically 8000)
    QWEN_MODEL_NAME = "qwen2.5vl:latest"  # or your specific qwen-vl checkpoint name
  
    FAISS_INDEX_PATH = "data/memory_index.faiss"

    