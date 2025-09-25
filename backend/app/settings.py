from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

# Resolve project root no matter where we run from:
# settings.py is at backend/app/settings.py -> go up two levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    # Pydantic v2 style config
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8")

    # App
    app_name: str = "ai-medical-chatbot"
    host: str = "0.0.0.0"
    port: int = 8000

    # RAG
    chroma_dir: Path = Path("/chroma_store")  # will be mounted by Docker later
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "medical_knowledge"

    # GenAI - FIXED: Use correct Gemini model name
    gemini_api_key: str | None = Field(default=None, validation_alias=AliasChoices('GEMINI_API_KEY','GOOGLE_API_KEY'))
    gemini_model: str = "gemini-1.5-flash"  # Use the standard Gemini API model name

    # CORS / Frontend
    frontend_origin: str = "http://localhost:5173"

settings = Settings()