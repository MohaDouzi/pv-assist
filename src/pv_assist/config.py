from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PV Assist"
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    debug: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pv_assist_documents"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    llm_provider: Literal["groq", "mistral"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    mistral_api_key: SecretStr | None = None
    mistral_model: str = "mistral-small-latest"
    min_retrieval_score: float = 0.35
    chunk_size: int = 700
    chunk_overlap: int = 150
    max_upload_size_bytes: int = 20 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap cannot be negative")
        return value

    @field_validator("min_retrieval_score")
    @classmethod
    def validate_min_retrieval_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("min_retrieval_score must be between 0 and 1")
        return value


settings = Settings()
