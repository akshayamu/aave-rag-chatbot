# Aave RAG Chatbot v2 – Zero Trace

A privacy-focused chatbot for answering questions about Aave using Retrieval-Augmented Generation (RAG).

## Features
- Zero-trace architecture (no external data storage)
- Uses Ollama for embeddings
- Uses Groq API for LLM inference
- Attention highlighting for best matching sentences
- Confidence scoring
- Streamlit UI

## Prerequisites
1. Python 3.8+
2. Ollama installed and running (https://ollama.com/)
3. Groq API key (https://console.groq.com/)

## Setup Instructions

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt