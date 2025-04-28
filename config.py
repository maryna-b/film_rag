"""
Configuration settings for the Film RAG system.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM configuration
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Vector database configuration
CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")

# RAG configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 3

# Recommendation configuration
SIMILARITY_THRESHOLD = 0.75
MAX_RECOMMENDATIONS = 3
