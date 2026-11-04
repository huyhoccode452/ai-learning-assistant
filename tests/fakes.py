"""Test doubles: no network, fully deterministic."""

import hashlib
import math
import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from app.config import Settings


def make_settings(**overrides: object) -> Settings:
    """Settings that ignore .env; small embedding_dim keeps tests fast."""
    values: dict[str, object] = {
        "gemini_api_key": "test-key-not-real-123",
        "gemini_embedding_model": "fake-embedding-model",
        "embedding_dim": 64,
        "embed_max_retries": 3,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts that share words get similar vectors.

    Good enough to test retrieval ordering without any network call. Implements the
    app.embeddings.Embedder protocol (same vector for documents and queries).
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            raise ValueError(f"no words in {text!r}")
        return [x / norm for x in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeModels:
    """Stands in for `genai.Client().models`. Records calls; can fail on a script.

    `script` is consumed one item per embed_content call: an Exception is raised, None means
    "behave normally". Texts must start with "t<number>"; the vector is one-hot at that
    number (scaled by 5, i.e. NOT unit length) so tests can verify order and normalization.
    """

    def __init__(self, dim: int = 8, script: list[Exception | None] | None = None,
                 token_counts: dict[int, int] | None = None, drop_last: bool = False) -> None:
        self.dim = dim
        self.drop_last = drop_last  # simulate an API that returns one embedding too few
        self.script = list(script or [])
        self.token_counts = token_counts or {}
        self.embed_calls: list[dict[str, Any]] = []
        self.count_calls: list[str] = []

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> SimpleNamespace:
        self.embed_calls.append(
            {"model": model, "contents": list(contents), "task_type": config.task_type,
             "dim": config.output_dimensionality}
        )
        if self.script:
            outcome = self.script.pop(0)
            if outcome is not None:
                raise outcome
        vectors = []
        for text in contents:
            vector = [0.0] * self.dim
            vector[int(re.match(r"t(\d+)", text).group(1)) % self.dim] = 5.0  # type: ignore[union-attr]
            vectors.append(SimpleNamespace(values=vector))
        return SimpleNamespace(embeddings=vectors[:-1] if self.drop_last else vectors)

    def count_tokens(self, *, model: str, contents: str) -> SimpleNamespace:
        self.count_calls.append(contents)
        return SimpleNamespace(total_tokens=self.token_counts.get(len(contents), len(contents) // 4))


class FakeGenaiClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models
