"""
Central config loader. Every other module imports settings from here
so there's exactly one place that reads environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    VECTOR_DB: str = os.getenv("VECTOR_DB", "qdrant")

    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "textbooks")

    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "textbooks")

    # Gemini vision model — used for OCR (reads the question out of the photo)
    OCR_MODEL: str = os.getenv("OCR_MODEL", "gemini-flash-latest")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Gemini text model — used for the Socratic tutor
    SOCRATIC_MODEL: str = os.getenv("SOCRATIC_MODEL", "gemini-flash-latest")

    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", "8000"))


settings = Settings()

if not settings.GEMINI_API_KEY:
    # Fail loud at import time in dev — better than a confusing 401 mid-demo.
    print("[WARNING] GEMINI_API_KEY is not set. Set it in your .env file.")
