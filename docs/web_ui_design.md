# Web UI design (shared spec)

The plain HTML/CSS/JS web UI replaces nothing: the Streamlit app stays. It must offer the same flows (see `frontend/`) and talk to the same API. Every agent working on the web UI follows this file; if something is missing here, copy the behaviour and the Vietnamese texts of the Streamlit app.

## 1. Style
- Calm, readable study tool. White cards on a light grey page, 1px borders, 8px radius, no shadows except the mobile drawer, no gradients, no animations beyond 150ms colour/opacity transitions. Light theme only.
- All user-facing text is Vietnamese (reuse Streamlit strings); code and identifiers are English.

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#F7F8FA` | page background |
| `--surface` | `#FFFFFF` | cards, sidebar, inputs |
| `--border` | `#E2E5EA` | borders, dividers |
| `--text` / `--muted` | `#1F2937` / `#6B7280` | body text / captions |
| `--primary` / `--primary-hover` | `#2563EB` / `#1D4ED8` | primary buttons, active tab, links, focus ring |
| `--success` on `--success-bg` | `#15803D` on `#DCFCE7` | "Sẵn sàng", "Được trích dẫn", "Đã xác minh", correct answer |
| `--warning` on `--warning-bg` | `#B45309` on `#FEF3C7` | "Đang xử lý", timeouts, truncated, `saved: false` |
| `--danger` on `--danger-bg` | `#B91C1C` on `#FEE2E2` | "Lỗi", errors, "Tài liệu đã bị xóa", wrong answer |
| `--info` on `--info-bg` | `#1E40AF` on `#DBEAFE` | insufficient-evidence fallback, "Đang mở" |
| `--neutral` on `--neutral-bg` | `#4B5563` on `#F3F4F6` | "Chỉ truy xuất", "Chưa xác minh" |

- Font: `system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif` (covers Vietnamese); monospace `ui-monospace, Consolas, monospace` only for scores/latency. Sizes 13 (captions), 14 (UI), 16 (body, chat, inputs: avoids iOS zoom), 20 (page title); line-height 1.5. Spacing scale 4/8/12/16/24px.
- Accessibility: visible focus ring (`2px solid --primary`), real `<button>`/`<label>`, `aria-live="polite"` on the notice area and chat log, status never shown by colour alone (badge text always present), touch targets ≥ 40px.

## 2. Layout
- **Laptop (≥ 900px):** header bar (title "AI Learning Assistant" + health status). Below it two columns: left sidebar 320px (scrolls on its own) with, in order: upload, notice, document list, chat scope checkboxes, "Cuộc trò chuyện mới", history, footer note. Right main area (max width 860px) with tabs "Hỏi đáp" / "Tóm tắt" / "Trắc nghiệm". In "Hỏi đáp": follow-up note on top, scrolling message list, input box + "Gửi" pinned at the bottom of the main area.
- **Phone (< 900px):** one column. Header gets a "Tài liệu" button that opens the sidebar as a full-height drawer over the page (close button + backdrop + Esc). Tabs become a sticky row under the header. Chat input is fixed to the bottom of the screen; sources and quiz cards take the full width. No horizontal scroll at 360px.

## 3. API contract (same origin, paths relative to `/`)
All errors: `{"error": {"code", "message", "details"?}}`. Show `error.message` (Vietnamese, max 500 chars); if the body is unusable, use the Streamlit fallback text for the status (`frontend/api_client.py`). 429: append the `Retry-After` seconds. Network failure / abort: Streamlit's connection and timeout messages. Timeouts with `AbortController`: 3s health, 10s list/get/delete, 60s upload and chat, 120s summary and quiz.

| Call | Request | Success | Notable errors |
|---|---|---|---|
| `GET /health` | – | `{status: "ok"\|"degraded", qdrant_reachable}` | network error = backend down banner + "Thử kết nối lại" |
| `POST /documents` | multipart field `file` (.pdf/.docx/.txt, ≤ 20 MB, checked client-side first) | `Document`; **202** new/processing, **200** already ready | 413, 415, 400 `ingestion.empty_file`, 422 `ingestion.*` (scanned/encrypted/corrupted) |
| `GET /documents` | – | `Document[]`, newest first | – |
| `DELETE /documents/{doc_id}` | – | `{doc_id, deleted_chunks}` | 409 `document_processing`, 404 |
| `POST /chat` | JSON `{question (1–2000), doc_ids (1–50), conversation_id?}` | `{answer, sources[], used_fallback, best_score, truncated, latency{embed_ms,search_ms,llm_ms,total_ms}, conversation_id, saved}` | 404 `conversation_not_found` (reset id), 409 `document_not_ready`/`no_documents_ready`, 429, 503, 504 |
| `POST /documents/{doc_id}/summary` | JSON `{style: "short"\|"detailed", refresh}` | `{summary, style, num_chunks_used, num_chunks_total, truncated, warning, llm_calls, cached, latency_ms}` | 409, 429, 502–504 |
| `POST /documents/{doc_id}/quiz` | JSON `{num_questions 1–10, verify, refresh, topic? ≤200, page_from?, page_to?}` (omit nulls) | `{questions[], requested, returned, dropped{reason:n}, message, warnings[], verified, verifier_decisions[], llm_calls, cached, latency_ms}` | 422 `generation.empty_selection`, 502 `generation.malformed_output` |
| `GET /conversations?limit=10&offset=0` | limit 1–100 | `{items[], total, limit, offset}` | – |
| `GET /conversations/{id}` | – | item + `messages[]` | 404 |
| `DELETE /conversations/{id}` | – | `{conversation_id, deleted_messages}` | 404 |

- `Document`: `doc_id, filename, status ("processing"|"ready"|"failed"), error_code, error_message, num_pages|null, num_chunks|null, created_at`.
- `Source`: `number, doc_id, filename, page|null, chunk_index, score, excerpt, cited` (+ `document_exists` in history).
- Quiz question: `id, question, options[4], answer_index, explanation, verified|null, source{chunk_index, page, filename, quote}`.
- Conversation item: `id, title|null, doc_ids[], missing_doc_ids[], created_at, updated_at, message_count`; message: `role, content, created_at, sources[], used_fallback, best_score, truncated, latency` (assistant fields null for user messages).
- Polling: while any document is `processing`, `GET /documents` every 2s; per document show "slow" warning after 30s and stop after 180s with a "Kiểm tra lại" button.
- Fallback: when `used_fallback` is true show the info box "Không đủ bằng chứng trong tài liệu để trả lời." + answer + `best_score` caption; no sources.
- Upload status 200 vs 202, `saved: false`, `cached`, deleted-document warnings: same texts as Streamlit.

## 4. Files
- `web/index.html`: static skeleton (header, sidebar, tabs, empty containers, templates if needed). Links `styles.css` and `app.js` (`<script src="app.js" defer>`). No inline `<script>`, `<style>`, `style=""` or `on*=` attributes.
- `web/styles.css`: tokens from section 1 as CSS variables on `:root`, then layout, components, one `@media (max-width: 899px)` block.
- `web/app.js`: one file, sections in this order: config/constants, `api` (fetch wrapper + error mapping), `state`, DOM helpers (`el(tag, {className, text, attrs})`), renderers (documents, chat, sources, summary, quiz, history), event wiring, `init()`.
- Plain HTML/CSS/JS only: no framework, no CDN, no npm, no build step, no external fonts or icons (use text or inline Unicode).

## 5. Security rules
- Everything from the server (filenames, answers, excerpts, quotes, summaries, titles, error messages) is displayed with `textContent` or `createTextNode`. Never pass a non-literal string to `innerHTML`, `outerHTML`, `insertAdjacentHTML` or `document.write`. No `eval`, `new Function`, or string `setTimeout`.
- No Markdown parser: answers and summaries are shown as plain text with `white-space: pre-wrap` (citations `[n]` stay readable). Accepted limitation.
- Build DOM with `createElement`; set attributes with `setAttribute` only on literal names; never set `href`/`src` from server data.
- `index.html` sets `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'">`, which works because there is no inline code.
- IDs in URLs are passed through `encodeURIComponent`. Nothing from the server is stored in `localStorage`.

## 6. Mounting
- In `create_app()` (`app/main.py`), after the routes: `app.mount("/ui", StaticFiles(directory=<project_root>/web, html=True), name="ui")`, with the path resolved from `__file__`, not the working directory.
- Open `http://127.0.0.1:8000/ui/`. The UI calls the API with absolute same-origin paths (`/documents`, `/chat`, …), so CORS needs no change and `Retry-After` is readable.
- A test checks that `GET /ui/` returns `index.html` and that the API routes still work.
