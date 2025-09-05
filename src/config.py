# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ollama configuration - using a smaller embedding model for faster processing
OLLAMA_EMBEDDING_MODEL = "all-minilm"  # sentence-transformers based
OLLAMA_BASE_URL = "http://localhost:11434"

# Groq configuration - updated model (old one was decommissioned)
# Options: "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama-3.2-1b-preview", "llama-3.2-3b-preview"
GROQ_LLM_MODEL = "llama-3.1-8b-instant"
