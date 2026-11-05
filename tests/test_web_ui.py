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
