# Web UI design (shared spec, v2)

The plain HTML/CSS/JS web UI (`web/`, served at `/ui/`) talks to the same API as the Streamlit app, which stays. This file is the **contract between the files** `web/index.html`, `web/styles.css`, `web/markdown.js` and `web/app.js`. Each file is written against sections 3 and 4; if a name is not listed there, do not invent it silently: change this file first.

Look and feel: calm study tool in the spirit of NotebookLM / Perplexity: lots of white space, readable text, one accent colour, citations that are visible but never louder than the answer. All user-facing text is Vietnamese; code and identifiers are English.

## 1. Design tokens

All values live on `:root` as CSS custom properties. Components use tokens only (no hex values outside the two token blocks). Dark mode is automatic through `@media (prefers-color-scheme: dark)` which redefines the same variables (no toggle, no JS). `:root` also sets `color-scheme: light dark`.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#FAFAF9` | `#0F1115` | page |
| `--surface` | `#FFFFFF` | `#171A21` | cards, inputs, popovers, composer |
| `--surface-2` | `#F5F5F4` | `#13161C` | sidebar, hover fills, code background |
| `--border` | `#E7E5E4` | `#262B35` | hairlines |
| `--border-strong` | `#D6D3D1` | `#3A404D` | input borders, dropzone |
| `--text` | `#1C1917` | `#E7E9EE` | body |
| `--text-2` | `#44403C` | `#B4BAC6` | secondary text |
| `--muted` | `#78716C` | `#8B93A1` | captions, meta (≥ 4.5:1 on `--surface`) |
| `--accent` | `#3F5EFB` | `#8EA0FF` | the only accent: primary button, active tab, links, focus ring, citations |
| `--accent-hover` | `#3049D8` | `#A9B6FF` | hover |
| `--accent-soft` | `#EEF1FF` | `#1E2440` | citation pill, selected rows, info background |
| `--on-accent` | `#FFFFFF` | `#0B1020` | text on `--accent` |
| `--success` / `--success-bg` | `#166534` / `#DCFCE7` | `#86EFAC` / `#12301F` | ready, correct, cited |
| `--warning` / `--warning-bg` | `#92400E` / `#FEF3C7` | `#FCD34D` / `#3A2E0E` | processing, timeouts, `saved: false` |
| `--danger` / `--danger-bg` | `#B91C1C` / `#FEE2E2` | `#FCA5A5` / `#3B1518` | errors, wrong answer, deleted |
| `--info` / `--info-bg` | `#3049D8` / `#EEF1FF` | `#A9B6FF` / `#1E2440` | insufficient-evidence card, "Đang mở" |

Other tokens (same in both themes): radius `--radius-sm: 6px`, `--radius: 10px`, `--radius-lg: 16px`, `--radius-pill: 999px`; spacing `--space-1…8` = 4, 8, 12, 16, 20, 24, 32, 48 px; shadows `--shadow-sm` (cards, composer), `--shadow-md` (popover), `--shadow-lg` (mobile drawer); fonts `--font-sans: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif` (covers Vietnamese) and `--font-mono: ui-monospace, Consolas, monospace` (scores, latency, code only); sizes `--fs-xs 12`, `--fs-sm 13`, `--fs-base 15`, `--fs-md 16` (inputs and chat text: avoids iOS zoom), `--fs-lg 20`, `--fs-xl 28`; line-height 1.5 (answers 1.65); `--sidebar-w: 300px`, `--content-w: 800px`, `--topbar-h: 56px`, `--t-fast: 150ms`.

Rules: status is never shown by colour alone (badge text is always present); visible focus ring `2px solid var(--accent)` with 2px offset on every interactive element; touch targets ≥ 40px on phones; transitions ≤ 150ms on colour/opacity/transform; the typing-dots animation and any other motion are disabled under `prefers-reduced-motion: reduce`; `[hidden]` always wins (`[hidden] { display: none !important; }`); no inline `style=` anywhere.

## 2. Layout

- **Page**: `body` is a column of height `100dvh` (fallback `100vh`): `.topbar`, optional `#backend-down` banner, then `.layout` filling the rest. Only inner areas scroll, never the page.
- **Top bar** (`.topbar`, 56px, `--surface`, bottom hairline): left `#open-drawer` (phones only) and the brand (`.brand-mark` + `h1.brand-name` "AI Learning Assistant"); right the status chip `#health` = a coloured dot + short text (dot colour AND text).
- **Laptop (≥ 900px)**: `.layout` is two columns. Left `#sidebar` 300px on `--surface-2`, own scroll, two sections: **Tài liệu** (dropzone, notice area, document list, "Hỏi trong" scope checkboxes) and **Lịch sử** (heading row with the "+ Cuộc trò chuyện mới" button, list, "Xem thêm"), and a one-line footnote. Right `.main`: tabs "Hỏi đáp / Tóm tắt / Trắc nghiệm" as a slim underlined row, then the active panel.
- **Chat panel**: `#chat-scroll` scrolls; inside it `.chat-inner` (max `--content-w` 800px, centred, horizontal padding 16–24px) holds `#empty-state` (shown only before the first message) and `#chat-log`. Below, **pinned to the bottom of the panel**, the `.composer` holds the rounded input box (`#chat-form.composer-box`: auto-growing-looking textarea `#question` + round send button `#send-btn`) and the one-line note `#followup-note` (small `(i)` icon + short text; the full explanation is in its `title`). Messages: user = right-aligned bubble (`--accent-soft`); assistant = no bubble, plain text column on the page background.
- **Summary / Quiz panels**: `.panel-inner` (max 800px, centred) with stacked fields, a primary and a secondary button, and the result below as a card.
- **Phone (< 900px)**: one column. `#open-drawer` ("☰ Tài liệu") appears; `#sidebar` becomes a fixed full-height drawer (`min(320px, 90vw)`, `--shadow-lg`, closed state `visibility: hidden` + `translateX(-100%)`) over `#backdrop`; tabs stay a sticky row under the top bar; the composer stays at the bottom of the screen (it is the last flex child, not `position: fixed`); source cards and quiz cards are full width; no horizontal scroll at 360px (`overflow-wrap: anywhere` on user/server text). Citation popover becomes a bottom sheet (see 3.7).
- **Empty state** (`#empty-state`): centred column with a mark, `h2.empty-title` "Hỏi bất cứ điều gì về tài liệu của bạn", `p#empty-sub` (one line; JS replaces it by the reason chat is disabled, e.g. "Hãy tải lên một tài liệu và đợi trạng thái “Sẵn sàng”"), and `#suggestions`: four clickable `button.suggestion` (2×2 grid on laptop, 1 column on phone).
- **Dropzone** (`label#dropzone`): dashed border, icon, "Kéo thả tệp vào đây hoặc bấm để chọn", hint "PDF, DOCX, TXT · tối đa 20 MB". Dragging a file over it adds `.is-dragover`; while uploading `.is-busy`. The file input is visually hidden but focusable; `#dropzone:focus-within` shows the focus ring.

## 3. DOM contract

`index.html` provides every **static** element in 3.1. `app.js` creates every **dynamic** element in 3.2–3.8 with exactly these tags and classes (`el(tag, {className, text, attrs})`). `styles.css` styles all of them. Anything not listed here is not part of the contract.

### 3.1 Static elements (ids in `index.html`)

| Id | Element | Notes |
|---|---|---|
| `open-drawer` | `button.btn.btn-ghost.drawer-toggle` | text "☰ Tài liệu"; only visible < 900px |
| `health` | `span.status` with children `span.status-dot` and `span#health-text.status-text`; `role="status"` | JS sets one tone class: `is-checking` (initial) / `is-ok` / `is-degraded` / `is-down`. Texts: "Đang kiểm tra…", "Hoạt động", "Qdrant không kết nối", "Mất kết nối" |
| `backend-down`, `backend-down-text`, `retry-health` | `div.banner` (hidden by default), `span`, `button.btn.btn-secondary.btn-sm` "Thử kết nối lại" | |
| `backdrop` | `div.backdrop[hidden]` | |
| `sidebar` | `aside.sidebar` | `aria-label="Tài liệu và lịch sử"` |
| `close-drawer` | `button.btn.btn-ghost.btn-sm.drawer-toggle` "Đóng" | inside `.sidebar-head`; phones only |
| `dropzone` | `label.dropzone[for=file-input]` | contains `span.dropzone-icon` (aria-hidden), `span.dropzone-title`, `span.dropzone-hint`, and the input |
| `file-input` | `input[type=file][accept=".pdf,.docx,.txt"].visually-hidden` | selecting or dropping a file starts the upload immediately (no separate upload button) |
| `notice` | `div.notice[aria-live=polite]` | holds at most one `.alert` |
| `doc-list`, `doc-empty`, `poll-info`, `recheck` | `ul.doc-list`, `p.side-empty[hidden]`, `div.caption`, `button.btn.btn-secondary.btn-sm[hidden]` "Kiểm tra lại" | |
| `scope-list`, `scope-empty` | `div.scope-list`, `p.side-empty` | under `h3.side-subtitle` "Hỏi trong" |
| `new-chat` | `button.btn.btn-secondary.btn-sm` "+ Cuộc trò chuyện mới" | in the heading row of the Lịch sử section |
| `history-list`, `history-empty`, `history-more` | `ul.history-list`, `p.side-empty[hidden]`, `button.btn.btn-ghost.btn-sm[hidden]` "Xem thêm" | |
| `tab-chat`, `tab-summary`, `tab-quiz` | `button.tab[role=tab][data-tab=chat\|summary\|quiz]` | active one has `.is-active` and `aria-selected="true"` |
| `panel-chat`, `panel-summary`, `panel-quiz` | `section.panel[role=tabpanel]` | inactive ones are `hidden`; `panel-chat` also has `.chat-panel` |
| `chat-scroll` | `div.chat-scroll` | the scroll container of the chat; contains one `div.chat-inner` |
| `empty-state`, `empty-sub`, `suggestions` | `section.empty-state`, `p.empty-sub`, `div.suggestions` with four `button.suggestion[type=button]` | suggestion texts below |
| `chat-log` | `div.chat-log[aria-live=polite]` | inside `.chat-inner`, after `#empty-state` |
| `chat-form`, `question`, `send-btn` | `form.composer-box`, `textarea[maxlength=2000][rows=1]`, `button.btn.btn-primary.btn-send[type=submit]` "Gửi" | form is inside `div.composer` |
| `followup-note` | `p.note` with `span.note-icon` (aria-hidden, text "i") and `span.note-text` | text: "Mỗi câu hỏi được trả lời độc lập; hãy nêu rõ chủ đề." and `title` carries the long explanation (earlier messages are not used to understand follow-ups; history is for reading only) |
| `summary-doc`, `summary-make`, `summary-again`, `summary-out` | `select`, `button.btn.btn-primary`, `button.btn.btn-secondary`, `div[aria-live=polite]` | style radios: `fieldset.segmented` with two `label.segment > input[type=radio][name=summary-style][value=short\|detailed] + span`; `short` is checked |
| `quiz-doc`, `quiz-count`, `quiz-from`, `quiz-to`, `quiz-topic`, `quiz-verify`, `quiz-make`, `quiz-new`, `quiz-out` | as today (select, number inputs, text, checkbox, `btn-primary`, `btn-secondary`, output `div[aria-live=polite]`) | the three numbers sit in `div.field-row` |

Suggestions (static text in `index.html`, generic so they fit any document; clicking one fills `#question` and submits): "Tài liệu này nói về chủ đề chính nào?", "Tóm tắt các ý chính của tài liệu.", "Liệt kê các khái niệm quan trọng và định nghĩa của chúng.", "Có những lưu ý hoặc hạn chế nào được nhắc đến?"

Generic helpers every file may rely on: `.visually-hidden`, `.caption` (13px `--muted`), `.mono`, `.row` (flex, wraps, gap 8), `.item-actions`, `.field` (label + control), `.label`, `.field-row`, `.check`, `.side-title`, `.side-subtitle`, `.side-section`, `.side-heading`, `.side-foot`.

### 3.2 State classes (only these)

`.is-active` (tab), `.is-open` (citation popover), `.is-highlight` (source card), `.is-busy` (dropzone while uploading), `.is-dragover` (dropzone), `.is-current` (history item that is open), `.is-correct` / `.is-wrong` (quiz option after checking), `.is-pending` (assistant message waiting for the model), `.is-left` / `.is-right` / `.is-top` (citation popover placement), status tones `is-checking|is-ok|is-degraded|is-down` on `#health`. `body.drawer-open` opens the phone drawer.

### 3.3 Shared components (created by `app.js`)

| Component | Markup |
|---|---|
| Buttons | `.btn` + one variant: `.btn-primary` (accent fill), `.btn-secondary` (outline), `.btn-ghost` (no border), `.btn-danger` (danger text/outline); size `.btn-sm`. `button:disabled` is dimmed, `cursor: not-allowed`. |
| Badge | `span.badge.badge-{success\|warning\|danger\|info\|neutral}` with text |
| Alert | `div.alert.alert-{success\|warning\|danger\|info\|neutral}[role=alert\|status]` with text (replaces the old `.box`) |

### 3.4 Documents and scope (sidebar)

```
li.doc-item
  div.doc-main
    span.doc-ext                 text "PDF" | "DOCX" | "TXT" (from the filename extension)
    div.doc-text
      div.doc-name               filename (single line, ellipsis, title attr = filename)
      div.doc-meta               span.badge.badge-* + span.doc-meta-text ("9 trang · 29 đoạn")
  div.doc-actions                button.btn.btn-ghost.btn-sm.doc-delete "Xóa"
  [div.alert.alert-warning]      slow-processing hint
  [div.doc-error.caption]        failed documents
  [div.confirm]                  div.alert.alert-warning + div.item-actions(button.btn.btn-danger.btn-sm "Xác nhận xóa", button.btn.btn-secondary.btn-sm "Hủy")
div.scope-item                   label[for] > input[type=checkbox] + span   (one per ready document, inside #scope-list)
```

### 3.5 History (sidebar)

```
li.history-item[.is-current]
  div.history-main
    div.history-title            span.history-title-text (max 2 lines, title attr = full title) + span.badge.badge-info "Đang mở" when current
    div.history-meta.caption     one span per segment ("05/11/2026, 18:30", "4 tin nhắn", "1 tài liệu") separated by " · "
  div.history-actions            button.btn.btn-ghost.btn-sm "Mở" (disabled when current), button.btn.btn-ghost.btn-sm "Xóa"
  [div.confirm]                  same as documents
```

### 3.6 Chat messages (inside `#chat-log`)

```
div.msg.msg-user > div.msg-bubble                      question text (pre-wrap)
div.msg.msg-assistant[.is-pending]
  div.msg-role                                          "Trợ lý"
  pending:   div.pending > span.typing-dots(3 × span) + span.pending-text
  answer:    div.md                                     built by renderMarkdown (3.9)
  fallback:  div.fallback-card > div.fallback-title "Không đủ bằng chứng trong tài liệu để trả lời." + div.md (server text) ; then p.caption (best_score explanation)
  [div.alert.alert-warning]                             truncated / saved:false
  details.sources                                       (collapsed by default; omitted when there are no sources, and always omitted for fallbacks)
    summary                                             "Nguồn (n)"
    [p.caption]                                         "Câu trả lời không trích dẫn nguồn nào; …"
    div.source-card#src-<msg>-<n>[.is-highlight]
      div.source-head
        span.source-num                                 "n"
        span.source-file                                filename
        span.source-page                                "Trang 5" | "Không có số trang"
        span.badge.badge-success "Được trích dẫn"  |  span.badge.badge-neutral "Chỉ truy xuất"
        [span.badge.badge-danger "Tài liệu đã bị xóa"]
        span.source-score.mono                          "0.738"
      p.source-excerpt                                  excerpt (pre-wrap)
  p.timing.caption.mono                                 latency line: one span per segment ("Thời gian: nhúng câu hỏi 120 ms", "tìm kiếm 40 ms", "mô hình 1.3 s", "tổng 1.5 s") separated by " · " text nodes, so it only wraps between segments
```

### 3.7 Citations

Created by `app.js` through the `onCitation` callback (3.9), once per `[n]` whose number is a known source:

```
span.cite-wrap
  button.cite-pill[type=button][aria-label="Nguồn n"][aria-expanded=false|true][aria-controls=<popover id>]   text "n"
  span.cite-popover#<popover id>[role=dialog][aria-label="Nguồn n"]
    span.cite-pop-head: span.cite-pop-file + span.cite-pop-page
    span.cite-pop-excerpt
    button.cite-pop-jump.btn.btn-ghost.btn-sm "Xem thẻ nguồn"
```

Styling: `.cite-pill` is a small superscript-style numeral (≈ 11–12px, `--accent` on `--accent-soft`, pill radius, min touch area ≥ 24px tall on laptop and ≥ 28px on phones through padding/`::after`; it must not break the line height). `.cite-wrap` is `position: relative; display: inline`. `.cite-popover` is hidden by default and shown by `.cite-wrap:hover`, `.cite-wrap:focus-within` and `.cite-wrap.is-open`; it opens below the pill, 320px wide (max 90vw), `--surface`, `--shadow-md`, 1px border. There must be no gap between pill and popover (hover bridge via `::before`). `.is-left` anchors its left edge to the pill, `.is-right` its right edge, `.is-top` opens upwards (JS adds them when the default position would overflow). Below 600px it is a bottom sheet: `position: fixed; left: 12px; right: 12px; bottom: 96px`. `Esc`, a click outside, or a second click on the pill closes it. "Xem thẻ nguồn" opens `details.sources`, scrolls to the card and gives it `.is-highlight`.

### 3.8 Summary and quiz output

```
#summary-out:  div.summary-card > div.md (+ [div.alert.alert-warning]) ; p.caption.summary-meta ("Dùng 29/29 đoạn · lấy từ kết quả đã lưu …")
#quiz-out:     p.caption.quiz-meta ; [div.alert.alert-info|warning] ; p.caption (dropped) ;
               form > div.quiz-q > fieldset > legend.quiz-legend + label.quiz-option[.is-correct|.is-wrong] > input[type=radio] + span
                                  + div.quiz-result > div.alert.alert-* + div.quiz-answer + div.quiz-explain + div.caption.quiz-source + blockquote.quote
               div.row > button.btn.btn-primary "Kiểm tra" + button.btn.btn-secondary "Làm lại bộ câu hỏi này" ; h3.quiz-score "Điểm: 2/3" (after checking it also gets `.alert.alert-success|danger|info` for all / none / some correct, and is scrolled into view)
```

### 3.9 Markdown: `web/markdown.js`

Plain script (no module, no dependency), loaded **before** `app.js` (`<script src="markdown.js" defer>` then `<script src="app.js" defer>`). It exposes exactly one global:

```js
window.renderMarkdown(text, { onCitation }) -> DocumentFragment
```

- `text`: any value; non-strings are treated as `""`. Always returns a fragment (empty for empty input). `onCitation(number)` is optional; it receives an integer and returns a DOM `Node` to insert in place of `[number]`, or `null`/`undefined` to leave the literal text `[number]`.
- **Block syntax** (output tags and classes): paragraph `p.md-p`; a single newline inside a paragraph `br.md-br`; bullet list lines starting with `- `, `* ` or `• ` → `ul.md-ul > li.md-li`; numbered lines `1. ` or `1) ` → `ol.md-ol > li.md-li`; one nested level (a list line indented by ≥ 2 spaces or a tab under an item) → a nested `ul/ol` inside that `li`; deeper indentation is flattened to level 2; headings `#` … `######` + space → `h4.md-h` (all levels look the same); fenced code ```` ``` ```` … ```` ``` ```` → `pre.md-pre > code.md-code` (text only; an unclosed fence runs to the end). Anything else (tables, quotes, HTML, `[text](url)` links, images) is plain text. **No `a`, `img` or `script` element is ever produced.**
- **Inline syntax**: `**bold**` → `strong.md-strong`; `*italic*` → `em.md-em` (the opening `*` must be followed by a non-space and the closing `*` preceded by a non-space, so `2 * 3 * 4` stays literal); `` `code` `` → `code.md-code` (nothing inside is parsed, not even `[n]`); `[n]` (digits only, 1–3 digits) → the node from `onCitation`. `_`/`__` are **not** emphasis (identifiers like `random_page_cost` must survive). A backslash before `\ ` `` ` `` `*` `[` `]` makes it literal. Unclosed markers stay literal text. `[1, 2]` and `[a]` are not citations. Bold may contain italic, code and citations; italic may contain code and citations.
- **Safety (hard rules)**: build output only with `document.createElement`, `createTextNode`/`textContent` and `appendChild`/`DocumentFragment`; the words `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `eval` and `new Function` must not appear in the file. Linear-time scanner: no regular expression with nested quantifiers; a 200 KB input must render in under 200 ms.
- `app.js` uses it for assistant answers, the fallback text and summaries. Quiz questions, explanations, quotes, filenames, titles and excerpts stay plain `textContent`.

## 4. Behaviour contract for `app.js`

- API table, error handling, timeouts and polling (sections 5 and 6) are unchanged from v1 and unchanged in code.
- Upload starts when a file is chosen or dropped on `#dropzone` (drop sets `#file-input.files` and calls the same `uploadFile()`); `.is-busy` and `input.disabled` while uploading; client-side extension and size checks stay.
- `renderEmptyChat()` shows `#empty-state` and clears `#chat-log`; `updateChatAvailability()` keeps `#empty-sub` and the `disabled` state of the suggestion buttons in sync with `disabledReason()`. Sending a question or opening a conversation sets `#empty-state.hidden = true`.
- Clicking a suggestion puts its text in `#question` and submits the form (same path as typing). Enter sends, Shift+Enter inserts a newline; `#question` grows with its content up to ~6 lines by toggling a row count (`rows` attribute), never inline `style`.
- `scrollChatToEnd()` scrolls `#chat-scroll`.
- Citation popover placement and closing (3.7) are handled in `app.js`; it never sets `style` or `href`/`src` from server data.

## 5. API contract (same origin, paths relative to `/`)

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
- Polling: while any document is `processing`, `GET /documents` every 2s (the list is re-rendered only when it changed); per document show the "slow" warning after 30s and stop after 180s with a "Kiểm tra lại" button.
- Fallback: when `used_fallback` is true show the fallback card (3.6) + `best_score` caption; no sources.
- Upload status 200 vs 202, `saved: false`, `cached`, deleted-document warnings: same texts as Streamlit.

## 6. Files and ownership

| File | Content | Written by |
|---|---|---|
| `web/index.html` | static skeleton from 3.1; `<link rel="stylesheet" href="styles.css">`, then `<script src="markdown.js" defer>`, then `<script src="app.js" defer>`; favicon `<link rel="icon" href="data:image/svg+xml,…">` (URL-encoded inline SVG, no `#` characters, no external file); no inline `<script>`, `<style>`, `style=""` or `on*=` | Style agent |
| `web/styles.css` | tokens (section 1) as the first block, dark-mode block, then base, layout, components; one `@media (max-width: 899px)` block for the drawer/phone layout and one `@media (max-width: 599px)` block for the popover bottom sheet and 1-column grids | Style agent |
| `web/markdown.js` | section 3.9 | Markdown agent |
| `web/app.js` | config/constants, `api`, `state`, DOM helpers, renderers (documents, chat, sources, citations, summary, quiz, history), event wiring, `init()` | main agent |

Plain HTML/CSS/JS only: no framework, no CDN, no npm, no build step, no external fonts or icons (text and Unicode only). The Docker image copies `web/`, so changes need `docker compose up -d --build api`.

## 7. Security rules

- Everything from the server (filenames, answers, excerpts, quotes, summaries, titles, error messages) is displayed with `textContent`, `createTextNode`, or through `renderMarkdown` (which itself only creates elements and text nodes). Never pass a non-literal string to `innerHTML`, `outerHTML`, `insertAdjacentHTML` or `document.write`. No `eval`, `new Function`, or string `setTimeout`.
- Markdown never produces `a`, `img`, `script`, `iframe` or attributes derived from the text. Set attributes with `setAttribute` only on literal names; never set `href`/`src`/`style` from server data. `id`/`value`/`for` may carry ids generated by the app or the server's hex ids.
- `index.html` sets `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'">`. The only addition to v1 is `img-src data:`, needed for the inline SVG favicon (no server data ever reaches an image).
- IDs in URLs are passed through `encodeURIComponent`. Nothing from the server is stored in `localStorage`.

## 8. Mounting

- In `create_app()` (`app/main.py`), after the routes: `app.mount("/ui", StaticFiles(directory=<project_root>/web, html=True), name="ui")`, with the path resolved from `__file__`. Open `http://127.0.0.1:8000/ui/`. The UI calls the API with absolute same-origin paths, so CORS needs no change.
- `tests/test_web_ui.py` checks: `/ui/` and the three assets are served, path traversal is 404, the API still works, no inline code or dangerous sinks in `index.html`, `app.js` and `markdown.js`, the exact CSP, the script order, the favicon, and that every static id of 3.1 exists in `index.html`.
