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
    assert "--accent" in css.text and "fetch(" in js.text
    markdown = client.get("/ui/markdown.js")
    assert markdown.status_code == 200 and "renderMarkdown" in markdown.text


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


@pytest.mark.parametrize("name", ["app.js", "markdown.js"])
def test_scripts_have_no_dangerous_sinks(name: str) -> None:
    js = (WEB / name).read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in js, f"forbidden sink in {name}: {sink}"
    assert not re.search(r"setTimeout\(\s*[\"'`]", js), "string setTimeout"
    assert not re.search(r"\.(href|src|style)\s*=", js), "href/src/style must never be assigned"


@pytest.mark.parametrize("path", ["/ui/../.env", "/ui/%2e%2e/.env", "/ui/..%2f..%2f.env", "/ui/%2e%2e%2fapp/main.py"])
def test_ui_mount_does_not_expose_files_outside_web(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert "GEMINI_API_KEY" not in response.text and "create_app" not in response.text


def test_index_csp_is_exact() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    csp = "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'"
    assert f'http-equiv="Content-Security-Policy" content="{csp}"' in html


def test_app_js_keeps_review_fixes() -> None:
    """Cheap guards for bugs found in the web UI review (KI-30); real behaviour was checked in a browser."""
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "Math.min(100" in js, "history reload must stay within the API limit of 100"
    assert "state.busy.has(id)" in js, "re-rendering must not re-enable buttons with a request in flight"
    assert '$("empty-state").hidden = true' in js, "sending a question must hide the welcome screen, not clear the log"
    assert "state.processingSince.clear()" in js, "a new upload must get a fresh 180s polling window"
    assert js.count("if (state.asking) { warnAsking(); return; }") == 2, "no chat switching while a reply is pending"


# Static ids that index.html must provide (docs/web_ui_design.md section 3.1): the contract between index.html and app.js.
STATIC_IDS = [
    "open-drawer", "health", "health-text", "backend-down", "backend-down-text", "retry-health", "backdrop", "sidebar",
    "close-drawer", "dropzone", "file-input", "notice", "doc-list", "doc-empty", "poll-info", "recheck", "scope-list",
    "scope-empty", "new-chat", "history-list", "history-empty", "history-more", "tab-chat", "tab-summary", "tab-quiz",
    "panel-chat", "panel-summary", "panel-quiz", "chat-scroll", "empty-state", "empty-sub", "suggestions", "chat-log",
    "chat-form", "question", "send-btn", "followup-note", "summary-doc", "summary-make", "summary-again", "summary-out",
    "quiz-doc", "quiz-count", "quiz-from", "quiz-to", "quiz-topic", "quiz-verify", "quiz-make", "quiz-new", "quiz-out",
]


def test_index_provides_every_contract_id() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    missing = [i for i in STATIC_IDS if i not in ids]
    assert not missing, f"index.html lacks ids from the contract: {missing}"
    assert len(re.findall(r'class="[^"]*\bsuggestion\b', html)) == 4, "four suggestion buttons"
    # app.js must not look up a static element that the contract does not list
    js = (WEB / "app.js").read_text(encoding="utf-8")
    used = set(re.findall(r'\$\("([a-z][a-z0-9-]*)"\)', js))
    unknown = sorted(i for i in used if i not in STATIC_IDS)
    assert not unknown, f"app.js uses ids that are not in the contract: {unknown}"


def test_index_scripts_and_favicon() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.index('src="markdown.js"') < html.index('src="app.js"'), "markdown.js must load before app.js"
    assert re.search(r'<link[^>]+rel="icon"[^>]+href="data:image/svg\+xml,', html), "inline SVG favicon"
    assert "#" not in re.search(r'rel="icon"[^>]+href="([^"]+)"', html).group(1), "unescaped # breaks a data URI"
    assert "color-scheme" in html


def test_markdown_js_exposes_only_render_markdown() -> None:
    js = (WEB / "markdown.js").read_text(encoding="utf-8")
    assert "window.renderMarkdown" in js
    assert len(re.findall(r"\bwindow\.[A-Za-z_]+\s*=", js)) == 1, "markdown.js may only assign window.renderMarkdown"
    assert "createElement(\"a\")" not in js and "createElement('a')" not in js, "no links from Markdown"


def test_css_has_tokens_and_dark_mode() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    for token in ("--bg", "--surface", "--border", "--text", "--muted", "--accent", "--radius", "--shadow-md", "--font-sans"):
        assert token + ":" in css, f"missing design token {token}"
    assert "prefers-color-scheme: dark" in css
    assert "prefers-reduced-motion" in css
    assert css.count("@media (max-width: 899px)") == 1 and css.count("@media (max-width: 599px)") == 1
