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


# ---------------------------------------------------------------------- RAG fakes


class FakeStore:
    """In-memory stand-in for QdrantStore.search: honors top_k, score_threshold and doc_ids,
    and records every call so tests can assert what the pipeline passed through."""

    def __init__(self, chunks: Sequence[Any] = ()) -> None:
        self.chunks = list(chunks)
        self.search_calls: list[dict[str, Any]] = []

    def search(self, query_vector: Sequence[float], top_k: int, score_threshold: float | None = None,
               doc_ids: Sequence[str] | None = None) -> list[Any]:
        self.search_calls.append({
            "top_k": top_k,
            "score_threshold": score_threshold,
            "doc_ids": None if doc_ids is None else list(doc_ids),
        })
        if doc_ids is not None and len(doc_ids) == 0:
            return []
        hits = [c for c in self.chunks
                if (doc_ids is None or c.doc_id in doc_ids)
                and (score_threshold is None or c.score >= score_threshold)]
        return sorted(hits, key=lambda c: -c.score)[:top_k]


class FakeLLM:
    """Implements app.rag.LLM. `reply` is the text to return, or an Exception to raise."""

    def __init__(self, reply: "str | Exception" = "ok [1]", **result_fields: Any) -> None:
        self.reply = reply
        self.result_fields = result_fields
        self.calls: list[dict[str, Any]] = []

    def generate(self, system_instruction: str, prompt: str, *, response_schema: Any = None,
                 max_output_tokens: int | None = None) -> Any:
        from app.rag import LLMResult

        self.calls.append({"system": system_instruction, "prompt": prompt, "response_schema": response_schema,
                           "max_output_tokens": max_output_tokens})
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResult(text=self.reply, **self.result_fields)


class FakeGenerateModels:
    """Stands in for `genai.Client().models` for generate_content. `script` is consumed one item
    per call: an Exception is raised, anything else is returned as the response."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.script.pop(0) if self.script else make_response("ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_response(text: str | None = "ok", *, finish: str | None = "STOP", block_reason: str | None = None,
                  candidates: bool = True, usage: tuple[int, int, int | None] = (10, 5, None)) -> SimpleNamespace:
    """A minimal GenerateContentResponse look-alike (only the attributes GeminiLLM reads)."""
    from google.genai import types

    reason = types.FinishReason[finish] if finish else None
    prompt_feedback = SimpleNamespace(block_reason=types.BlockedReason[block_reason]) if block_reason else None
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=reason)] if candidates else None,
        prompt_feedback=prompt_feedback,
        usage_metadata=SimpleNamespace(
            prompt_token_count=usage[0], candidates_token_count=usage[1], thoughts_token_count=usage[2]
        ),
    )


class FakeVectorStore(FakeStore):
    """In-memory stand-in for the whole QdrantStore surface used by app/main.py.

    Indexed chunks become searchable with a constant `score`, so chat tests can exercise the real
    RagPipeline threshold logic. `fail_index` / `fail_delete` make the next calls raise.
    """

    def __init__(self, *, score: float = 0.9, reachable: bool = True) -> None:
        super().__init__([])
        self.score = score
        self.reachable = reachable
        self.indexed: dict[str, list[Any]] = {}
        self.index_calls: list[str] = []
        self.ensure_calls = 0
        self.fail_index: Exception | None = None
        self.fail_delete: Exception | None = None
        self.fail_read: Exception | None = None

    def _rebuild(self) -> None:
        from app.vectorstore import RetrievedChunk

        self.chunks = [
            RetrievedChunk(self.score, c.doc_id, c.filename, c.page, c.chunk_index, c.text)
            for chunks in self.indexed.values() for c in chunks
        ]

    def is_reachable(self) -> bool:
        return self.reachable

    def ensure_collection(self) -> bool:
        self.ensure_calls += 1
        return False

    def index_document(self, doc_id: str, chunks: Sequence[Any], vectors: Sequence[Sequence[float]]) -> int:
        self.index_calls.append(doc_id)
        if self.fail_index:
            raise self.fail_index
        self.indexed[doc_id] = list(chunks)
        self._rebuild()
        return len(chunks)

    def get_chunks(self, doc_id: str) -> list[Any]:
        from app.vectorstore import RetrievedChunk

        if self.fail_read:
            raise self.fail_read
        return sorted((RetrievedChunk(0.0, c.doc_id, c.filename, c.page, c.chunk_index, c.text)
                       for c in self.indexed.get(doc_id, [])), key=lambda c: c.chunk_index)

    def delete_document(self, doc_id: str) -> int:
        if self.fail_delete:
            raise self.fail_delete
        removed = len(self.indexed.pop(doc_id, []))
        self._rebuild()
        return removed


class ScriptedLLM:
    """Plays back one scripted reply per call: a string, an Exception to raise, or a callable(system, prompt, kwargs)
    returning a string. A call beyond the script fails the test, so 'zero LLM calls' can be asserted by an empty script."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate(self, system_instruction: str, prompt: str, *, response_schema: Any = None,
                 max_output_tokens: int | None = None) -> Any:
        from app.rag import LLMResult

        self.calls.append({"system": system_instruction, "prompt": prompt, "response_schema": response_schema,
                           "max_output_tokens": max_output_tokens})
        assert self.script, "unexpected extra LLM call"
        item = self.script.pop(0)
        if callable(item):
            item = item(system_instruction, prompt, self.calls[-1])
        if isinstance(item, Exception):
            raise item
        return LLMResult(text=item)


DOC = "a" * 16


def make_doc_chunk(index: int, text: str | None = None, page: Any = "auto", doc_id: str = DOC,
                   filename: str = "tai_lieu.pdf"):
    """A RetrievedChunk with a distinct, quotable fact. Chunk i says: tham_so_i has default value i*10 MB."""
    from app.vectorstore import RetrievedChunk

    body = text if text is not None else (
        f"Tham số tham_so_{index} có giá trị mặc định là {index * 10}MB và được dùng cho tác vụ số {index}. "
        f"Nếu đặt quá cao, hệ thống tiêu tốn nhiều bộ nhớ hơn mức cần thiết cho tham_so_{index}."
    )
    return RetrievedChunk(0.0, doc_id, filename, index // 2 + 1 if page == "auto" else page, index, body)  # page=None is allowed


def make_doc_chunks(count: int, **kwargs: Any) -> list[Any]:
    return [make_doc_chunk(i, **kwargs) for i in range(count)]
