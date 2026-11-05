"""Application settings loaded from .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed configuration. Values come from OS env vars first, then `.env`.

    Model names and the embedding dimension have no defaults on purpose: they
    must be chosen from a real `models.list()` call and set in `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",  # absolute, so cwd does not matter
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `EMBEDDING_DIM=` -> treated as unset, not as ""
        extra="ignore",
        hide_input_in_errors=True,  # validation errors must never echo the API key
    )

    # Gemini
    gemini_api_key: SecretStr  # required; SecretStr masks it in repr/str/logs
    gemini_llm_model: str | None = None
    gemini_embedding_model: str | None = None
    embedding_dim: int | None = Field(default=None, gt=0)
    # Retries per API call on 429 / transient 5xx (0 = never retry). Used for embeddings AND the LLM.
    embed_max_retries: int = Field(default=5, ge=0, le=10)
    # How much the LLM "thinks" before answering. None = the model's own default.
    # Lower = faster and cheaper (measured with scripts/bench_llm.py, see docs/known_issues.md).
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    # Qdrant. For Qdrant Cloud set QDRANT_URL to the cluster URL (https://<id>.<region>.cloud.qdrant.io:6333)
    # and QDRANT_API_KEY; a local Qdrant needs no key.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    # Chunking / retrieval
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=5, gt=0)
    # Cosine similarity in [-1, 1]. None = not calibrated yet (set after evaluation).
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)

    # Uploads / storage
    max_upload_mb: int = Field(default=20, gt=0)
    # Public demo. DEMO_MODE lowers the upload size (max_upload_mb becomes the smaller of the two values,
    # see _apply_demo_mode) and caps how many documents the whole instance may hold (there are no accounts).
    demo_mode: bool = False
    demo_max_upload_mb: int = Field(default=5, gt=0)
    demo_max_documents: int = Field(default=10, gt=0)
    sqlite_path: Path = Path("data/app.sqlite3")
    # Original files are stored here under generated names (never the user's file name).
    upload_dir: Path = Path("data/uploads")

    # Abuse protection for the endpoints that call Gemini (chat, summary, quiz, upload). 0 = no limit.
    daily_limit_per_ip: int = Field(default=0, ge=0)
    # How many reverse proxies sit in front of the app (Render, Cloudflare...). 0 = use the socket address.
    # Otherwise the client IP is the Nth entry from the RIGHT of X-Forwarded-For (the left ones are forgeable).
    trusted_proxy_hops: int = Field(default=0, ge=0, le=5)

    @field_validator("sqlite_path", "upload_dir")
    @classmethod
    def _resolve_relative_to_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _apply_demo_mode(self) -> "Settings":
        # One place: the middleware, validate_upload and the UI-facing limit all read max_upload_mb.
        if self.demo_mode:
            self.max_upload_mb = min(self.max_upload_mb, self.demo_max_upload_mb)
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (also usable as a FastAPI dependency)."""
    return Settings()
