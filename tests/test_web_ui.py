"""Tests for the plain HTML/CSS/JS web UI served at /ui/. No network, no Qdrant: all dependencies are fakes."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_api import build_env

WEB = Path(__file__).resolve().parent.parent / "web"


@pytest.fixture
def client(tmp_path: Path):
    env = build_env(tmp_path)
    with TestClient(env.app) as c:
        yield c


def test_ui_index_is_served(client: TestClient) -> None:
    for path in ("/ui/", "/ui/index.html"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AI Learning Assistant" in response.text
        assert "Content-Security-Policy" in response.text


def test_ui_static_assets_are_served(client: TestClient) -> None:
    css = client.get("/ui/styles.css")
    js = client.get("/ui/app.js")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "--primary" in css.text and "fetch(" in js.text


def test_ui_missing_file_is_404(client: TestClient) -> None:
    assert client.get("/ui/nope.js").status_code == 404


def test_api_routes_still_work(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200 and "status" in health.json()
    documents = client.get("/documents")
    assert documents.status_code == 200 and documents.json() == []
    unknown = client.get("/does-not-exist")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "not_found"


def test_index_has_no_inline_code() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline <script>"
    assert "<style" not in html
    assert not re.search(r"\sstyle\s*=", html), "inline style attribute"
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handler"
    assert "https://" not in html and "http://" not in html


def test_app_js_has_no_dangerous_sinks() -> None:
    js = (WEB / "app.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in js, f"forbidden sink in app.js: {sink}"
    assert not re.search(r"setTimeout\(\s*[\"'`]", js), "string setTimeout"


@pytest.mark.parametrize("path", ["/ui/../.env", "/ui/%2e%2e/.env", "/ui/..%2f..%2f.env", "/ui/%2e%2e%2fapp/main.py"])
def test_ui_mount_does_not_expose_files_outside_web(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert "GEMINI_API_KEY" not in response.text and "create_app" not in response.text


def test_index_csp_is_exact() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    csp = "default-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'"
    assert f'http-equiv="Content-Security-Policy" content="{csp}"' in html


def test_app_js_keeps_review_fixes() -> None:
    """Cheap guards for bugs found in the web UI review (KI-30); real behaviour was checked in a browser."""
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "Math.min(100" in js, "history reload must stay within the API limit of 100"
    assert "state.busy.has(id)" in js, "re-rendering must not re-enable buttons with a request in flight"
    assert 'classList.contains("empty-hint")' in js, "only the empty-chat hint may be cleared on send"
    assert "state.processingSince.clear()" in js, "a new upload must get a fresh 180s polling window"
    assert js.count("if (state.asking) { warnAsking(); return; }") == 2, "no chat switching while a reply is pending"
