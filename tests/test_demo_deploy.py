"""Public-demo deployment features: per-IP daily limit, DEMO_MODE, Qdrant Cloud key, PORT.

No network: every dependency is a fake (same fixtures as test_api.py).
"""

import runpy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.ratelimit as ratelimit
import app.vectorstore as vectorstore
from app.config import Settings
from app.ratelimit import DailyRateLimiter, client_ip
from app.vectorstore import QdrantStore
from tests.fakes import make_settings
from tests.test_api import MB, TEXT, build_env, error_of, ready_document, upload

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def client_for(tmp_path: Path, **settings):
    env = build_env(tmp_path, **settings)
    client = TestClient(env.app)
    client.__enter__()  # runs the lifespan; closed by the caller through env.close()
    env.client = client
    env.close = lambda: client.__exit__(None, None, None)
    return env


@pytest.fixture
def limited(tmp_path):
    env = client_for(tmp_path, daily_limit_per_ip=3)
    yield env
    env.close()


@pytest.fixture
def plain(tmp_path):
    env = client_for(tmp_path)
    yield env
    env.close()


def doc_text(n: int) -> bytes:
    return TEXT + f"\n\nDocument number {n}.".encode()


# =========================================================================== DailyRateLimiter


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_limiter_allows_up_to_the_limit_then_refuses_without_counting_the_refusals() -> None:
    limiter = DailyRateLimiter(2)
    assert [limiter.hit("1.1.1.1").allowed for _ in range(4)] == [True, True, False, False]
    assert limiter.hit("1.1.1.1").used == 2  # refused requests do not raise the counter
    assert limiter.hit("2.2.2.2").allowed  # another address has its own counter


def test_limiter_resets_when_the_utc_day_changes_and_reports_the_time_to_reset() -> None:
    clock = Clock(datetime(2026, 11, 5, 23, 0, 0, tzinfo=UTC))
    limiter = DailyRateLimiter(1, clock=clock)
    assert limiter.hit("ip").allowed
    refused = limiter.hit("ip")
    assert not refused.allowed and refused.reset_in_seconds == 3600
    clock.now = datetime(2026, 11, 6, 0, 0, 1, tzinfo=UTC)
    assert limiter.hit("ip").allowed


def test_limiter_refuses_new_addresses_when_the_table_is_full(monkeypatch) -> None:
    monkeypatch.setattr(ratelimit, "MAX_TRACKED_IPS", 2)
    limiter = DailyRateLimiter(5)
    assert limiter.hit("a").allowed and limiter.hit("b").allowed
    assert not limiter.hit("c").allowed  # fail closed: the table must not grow without bound
    assert limiter.hit("a").allowed  # addresses already tracked are unaffected


def test_limiter_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError):
        DailyRateLimiter(0)


# =========================================================================== client_ip


def scope(peer: str | None = "10.0.0.9", forwarded: str | None = None) -> dict:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    return {"client": (peer, 1234) if peer else None, "headers": headers}


def test_client_ip_ignores_the_forwarded_header_when_no_proxy_is_trusted() -> None:
    assert client_ip(scope(forwarded="6.6.6.6"), 0) == "10.0.0.9"
    assert client_ip(scope(peer=None), 0) == "unknown"


def test_client_ip_takes_the_entry_the_trusted_proxy_appended_not_a_forged_one() -> None:
    # The client sent "6.6.6.6"; our single proxy appended the address it really saw.
    assert client_ip(scope(forwarded="6.6.6.6, 203.0.113.7"), 1) == "203.0.113.7"
    assert client_ip(scope(forwarded="6.6.6.6, 203.0.113.7, 198.51.100.2"), 2) == "203.0.113.7"


def test_client_ip_does_not_trust_a_header_shorter_than_the_proxy_chain() -> None:
    assert client_ip(scope(forwarded="203.0.113.7"), 2) == "10.0.0.9"
    assert client_ip(scope(), 1) == "10.0.0.9"


# =========================================================================== rate limit through the API


def test_the_chat_limit_is_enforced_with_a_vietnamese_message_and_no_llm_call(limited) -> None:
    doc_id = ready_document(limited)
    body = {"question": "work_mem là gì?", "doc_ids": [doc_id]}
    # The upload above already used 1 of 3.
    assert limited.client.post("/chat", json=body).status_code == 200
    assert limited.client.post("/chat", json=body).status_code == 200
    calls_before = len(limited.llm.calls)

    response = limited.client.post("/chat", json=body)
    assert response.status_code == 429
    error = error_of(response)
    assert error["code"] == "rate_limit.daily_exceeded"
    assert "3 lượt" in error["message"] and "07:00 giờ Việt Nam" in error["message"]
    assert error["details"]["limit"] == 3 and 0 < error["details"]["reset_in_seconds"] <= 86400
    assert "retry-after" not in response.headers  # the UIs would print "wait 40000 seconds"
    assert len(limited.llm.calls) == calls_before


@pytest.mark.parametrize("path", [
    "/chat", "/documents", "/documents/" + "a" * 16 + "/summary", "/documents/" + "a" * 16 + "/quiz",
])
def test_every_gemini_endpoint_is_refused_once_the_limit_is_used_up(limited, path: str) -> None:
    for _ in range(3):
        limited.client.get("/documents")  # reading is free and must not count
        assert limited.client.post("/chat", json={"question": "x"}).status_code == 409  # counted: no docs ready yet
    kwargs = {"files": {"file": ("a.txt", TEXT, "text/plain")}} if path == "/documents" else {"json": {"question": "x"}}
    response = limited.client.post(path, **kwargs)
    assert response.status_code == 429 and error_of(response)["code"] == "rate_limit.daily_exceeded"
    assert limited.db.list_documents() == []  # a refused upload stored nothing


def test_reading_endpoints_are_never_limited(limited) -> None:
    for _ in range(10):
        assert limited.client.get("/documents").status_code == 200
        assert limited.client.get("/conversations").status_code == 200
        assert limited.client.get("/health").status_code == 200


def test_a_limit_of_zero_means_unlimited(tmp_path) -> None:
    env = client_for(tmp_path, daily_limit_per_ip=0)
    try:
        for _ in range(8):
            assert env.client.post("/chat", json={"question": "x"}).status_code == 409
    finally:
        env.close()


def test_addresses_behind_a_trusted_proxy_have_separate_counters_and_a_forged_header_does_not_help(tmp_path) -> None:
    env = client_for(tmp_path, daily_limit_per_ip=1, trusted_proxy_hops=1)
    try:
        def ask(real_ip: str, forged: str = "9.9.9.9"):
            return env.client.post("/chat", json={"question": "x"},
                                   headers={"X-Forwarded-For": f"{forged}, {real_ip}"})

        assert ask("203.0.113.1").status_code == 409  # counted
        assert ask("203.0.113.1", forged="1.2.3.4").status_code == 429  # a new forged value changes nothing
        assert ask("203.0.113.2").status_code == 409  # a different real visitor has their own allowance
    finally:
        env.close()


def test_without_a_trusted_proxy_all_visitors_share_the_socket_address(tmp_path) -> None:
    env = client_for(tmp_path, daily_limit_per_ip=1, trusted_proxy_hops=0)
    try:
        assert env.client.post("/chat", json={"question": "x"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 409
        assert env.client.post("/chat", json={"question": "x"}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
    finally:
        env.close()


# =========================================================================== DEMO_MODE: settings


def test_demo_mode_off_leaves_the_upload_limit_alone() -> None:
    assert make_settings(max_upload_mb=20, demo_max_upload_mb=5).max_upload_mb == 20


def test_demo_mode_on_uses_the_smaller_upload_limit() -> None:
    assert make_settings(demo_mode=True, max_upload_mb=20, demo_max_upload_mb=5).max_upload_bytes == 5 * MB
    assert make_settings(demo_mode=True, max_upload_mb=2, demo_max_upload_mb=5).max_upload_mb == 2  # never raises it


def test_demo_settings_come_from_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_MAX_UPLOAD_MB", "3")
    monkeypatch.setenv("DEMO_MAX_DOCUMENTS", "4")
    monkeypatch.setenv("DAILY_LIMIT_PER_IP", "25")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    settings = make_settings()
    assert (settings.demo_mode, settings.max_upload_mb, settings.demo_max_documents) == (True, 3, 4)
    assert (settings.daily_limit_per_ip, settings.trusted_proxy_hops) == (25, 2)


def test_defaults_keep_local_use_unchanged() -> None:
    settings = make_settings()
    assert not settings.demo_mode and settings.daily_limit_per_ip == 0 and settings.trusted_proxy_hops == 0
    assert settings.qdrant_api_key is None


# =========================================================================== DEMO_MODE: API


def test_demo_mode_refuses_a_file_above_the_demo_size(tmp_path) -> None:
    env = client_for(tmp_path, demo_mode=True, max_upload_mb=20, demo_max_upload_mb=1)
    try:
        big = b"a" * (MB + 1024)
        response = upload(env, big)
        assert response.status_code == 413 and error_of(response)["code"] == "ingestion.file_too_large"
        assert upload(env, TEXT).status_code == 202  # a small file is fine
    finally:
        env.close()


def test_the_same_file_is_accepted_when_demo_mode_is_off(tmp_path) -> None:
    env = client_for(tmp_path, demo_mode=False, max_upload_mb=20, demo_max_upload_mb=1)
    try:
        assert upload(env, (b"word " * 300_000)[: MB + 1024]).status_code == 202
    finally:
        env.close()


def test_demo_mode_caps_the_number_of_documents_and_frees_a_slot_on_delete(tmp_path) -> None:
    env = client_for(tmp_path, demo_mode=True, demo_max_documents=2)
    try:
        first = ready_document(env, doc_text(1), "one.txt")
        ready_document(env, doc_text(2), "two.txt")
        response = upload(env, doc_text(3), "three.txt")
        assert response.status_code == 409
        error = error_of(response)
        assert error["code"] == "demo.document_limit" and "tối đa 2 tài liệu" in error["message"]
        assert len(env.db.list_documents()) == 2 and len(list(env.uploads.iterdir())) == 2  # nothing stored

        assert upload(env, doc_text(1), "one.txt").status_code == 200  # an already indexed file is not "new"
        assert env.client.delete(f"/documents/{first}").status_code == 200
        assert upload(env, doc_text(3), "three.txt").status_code == 202
    finally:
        env.close()


def test_failed_documents_do_not_use_up_the_demo_quota(tmp_path) -> None:
    env = client_for(tmp_path, demo_mode=True, demo_max_documents=1)
    try:
        env.db.claim_document("b" * 16, "broken.txt", None, 1)
        env.db.mark_failed("b" * 16, "embedding.rate_limited", "failed")
        assert upload(env, doc_text(1), "ok.txt").status_code == 202
    finally:
        env.close()


def test_without_demo_mode_the_number_of_documents_is_not_capped(tmp_path) -> None:
    env = client_for(tmp_path, demo_mode=False, demo_max_documents=1)
    try:
        for n in range(3):
            assert upload(env, doc_text(n), f"{n}.txt").status_code == 202
    finally:
        env.close()


# =========================================================================== Qdrant Cloud


class RecordingClient:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        RecordingClient.calls.append(kwargs)


@pytest.fixture
def recorder(monkeypatch):
    RecordingClient.calls = []
    monkeypatch.setattr(vectorstore, "QdrantClient", RecordingClient)
    return RecordingClient


def test_the_store_sends_the_cloud_url_and_api_key_to_the_client(recorder) -> None:
    settings = make_settings(qdrant_url="https://abc.eu-west.cloud.qdrant.io:6333", qdrant_api_key="s3cret-key")
    QdrantStore(settings)
    assert recorder.calls == [{"url": "https://abc.eu-west.cloud.qdrant.io:6333", "api_key": "s3cret-key", "timeout": 30}]


def test_a_local_qdrant_needs_no_key(recorder) -> None:
    QdrantStore(make_settings())
    assert recorder.calls[0]["api_key"] is None and recorder.calls[0]["url"] == "http://localhost:6333"


def test_an_empty_api_key_in_the_environment_counts_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "")
    assert make_settings().qdrant_api_key is None


def test_the_qdrant_api_key_is_masked_in_settings_output() -> None:
    settings = make_settings(qdrant_api_key="s3cret-key")
    assert "s3cret-key" not in repr(settings) and "s3cret-key" not in str(settings)


# =========================================================================== PORT / HOST


def test_the_port_comes_from_the_port_variable() -> None:
    assert main.server_address({"PORT": "10000"}) == ("127.0.0.1", 10000)


def test_defaults_are_port_8000_on_loopback_and_an_empty_port_counts_as_unset() -> None:
    assert main.server_address({}) == ("127.0.0.1", 8000)
    assert main.server_address({"PORT": ""}) == ("127.0.0.1", 8000)


def test_host_can_be_set_for_containers() -> None:
    assert main.server_address({"HOST": "0.0.0.0", "PORT": "8080"}) == ("0.0.0.0", 8080)


@pytest.mark.parametrize("bad", ["abc", "80.5", "0", "70000", "-1"])
def test_an_invalid_port_stops_the_server_with_a_clear_message(bad: str) -> None:
    with pytest.raises(SystemExit) as info:
        main.server_address({"PORT": bad})
    assert "PORT" in str(info.value)


def test_python_m_app_main_starts_uvicorn_on_the_requested_port(monkeypatch) -> None:
    started: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda application, **kwargs: started.update(app=application, **kwargs))
    monkeypatch.setenv("PORT", "12345")
    monkeypatch.delenv("HOST", raising=False)
    runpy.run_module("app.main", run_name="__main__")
    assert started["port"] == 12345 and started["host"] == "127.0.0.1"
    assert started["app"] is not None


def test_the_docker_image_runs_the_module_and_listens_on_all_interfaces() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "-m", "app.main"]' in dockerfile
    assert "HOST=0.0.0.0" in dockerfile  # without it the container would only listen on its own loopback


# =========================================================================== .env.example


def test_env_example_documents_every_new_variable() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("QDRANT_API_KEY", "DEMO_MODE", "DEMO_MAX_UPLOAD_MB", "DEMO_MAX_DOCUMENTS",
                 "DAILY_LIMIT_PER_IP", "TRUSTED_PROXY_HOPS", "PORT"):
        assert name in text, name
    fields = set(Settings.model_fields)
    for name in ("qdrant_api_key", "demo_mode", "demo_max_upload_mb", "demo_max_documents",
                 "daily_limit_per_ip", "trusted_proxy_hops"):
        assert name in fields
    assert "QDRANT_API_KEY=\n" in text.replace("\r\n", "\n")  # shipped empty: never a real key


# =========================================================================== site root (hosting health checks)


def test_get_root_redirects_to_the_web_ui_with_a_307(plain) -> None:
    response = plain.client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_following_the_root_redirect_serves_the_web_ui(plain) -> None:
    response = plain.client.get("/")  # TestClient follows redirects by default
    assert response.status_code == 200 and str(response.url).endswith("/ui/")
    assert "text/html" in response.headers["content-type"]


def test_head_root_answers_200_without_a_body_so_health_checks_pass(plain) -> None:
    response = plain.client.head("/", follow_redirects=False)
    assert response.status_code == 200
    assert response.content == b""


def test_the_root_routes_are_not_in_the_api_docs_and_do_not_shadow_the_api(plain) -> None:
    assert "/" not in plain.client.get("/openapi.json").json()["paths"]
    assert plain.client.get("/health").status_code == 200
    assert plain.client.get("/ui/").status_code == 200
    assert plain.client.get("/no-such-route").status_code == 404  # unknown paths still 404


def test_the_root_is_not_counted_by_the_daily_limit(limited) -> None:
    for _ in range(10):
        assert limited.client.head("/").status_code == 200
        assert limited.client.get("/", follow_redirects=False).status_code == 307
