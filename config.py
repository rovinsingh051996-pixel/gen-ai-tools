"""Configuration module for AI Studio application."""

import os
from pathlib import Path

# Application Settings
APP_NAME = "AI Studio"
APP_VERSION = "1.0.0"
LOG_FILE = "ai_app.log"

# LLM Configuration
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 30

# Embeddings Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Configuration
RAG_CHUNK_SIZE = 200
RAG_CHUNK_OVERLAP = 30
RAG_K_RESULTS = 3  # Number of documents to retrieve

# Input Validation
MIN_TEXT_LENGTH = 10
MIN_DESCRIPTION_LENGTH = 5
MIN_TOPIC_LENGTH = 3
MIN_QUIZ_QUESTIONS = 1
MAX_QUIZ_QUESTIONS = 10

# API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY', '')
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', '')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Feature Flags
ENABLE_RAG = True
ENABLE_LOGGING_FILE = True
ENABLE_DOTENV = True

def validate_config():
    """Validate that required configuration is present."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return True
