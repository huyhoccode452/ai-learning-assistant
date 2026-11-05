# Known issues

Issues we are aware of and have deliberately not fixed yet. Each entry says what we
verified, what we did not, and when to revisit. Update the status when one is resolved.

Last updated after conversation history. Last `pytest -q` run: 1172 passed, 29 skipped (the Qdrant-dependent tests, Qdrant was not running), 3 deselected (integration tests that call the real Gemini API).

| ID | Area | Issue | Status |
|----|------|-------|--------|
| KI-01 | ingestion | Repeated page headers/footers are indexed | Open; evaluation shows no harm but did not isolate it |
| KI-02 | ingestion | A heading can end a chunk, away from its body | Open |
| KI-03 | ingestion | PDF line structure (lists, tables) is flattened | Open, by design |
| KI-04 | ingestion | Partially scanned PDFs are accepted | Open |
| KI-05 | ingestion | No OCR for scanned PDFs | Out of scope |
| KI-06 | ingestion | DOCX: no page numbers, some content not read | Open |
| KI-07 | ingestion | Line-end hyphen re-joining can merge real compounds | Open |
| KI-08 | vectorstore | Re-indexing can leave stale chunks | Resolved in `QdrantStore.index_document` |
| KI-09 | api | Upload size limit is checked after the file is in memory | Resolved in `main.py` (middleware + bounded read) |
| KI-10 | tests | Some tests depend on a system font | Open |
| KI-11 | dependencies | `langchain-text-splitters` footprint and a known upstream bug | Decided: keep |
| KI-12 | embeddings | Free-tier throughput is ~100 texts per 70-85 s; embedding blocks | Handled in `main.py` (background job); throughput limit remains |
| KI-13 | vectorstore | `index_document` is delete-then-upsert, not atomic | Open, by design |
| KI-14 | retrieval | Cover/TOC chunks are retrieved as noise | Open; measured: in 9 of 18 prompts, top-1 for 2 absent-topic questions |
| KI-15 | vectorstore | `list_documents` makes one extra request per document | Open |
| KI-16 | embeddings | Token-limit pre-check rests on an assumption for short texts | Open |
| KI-17 | rag | `SCORE_THRESHOLD` is provisional; related-but-absent queries overlap the answerable range | Open; evaluation confirms the overlap, 0.65 kept |
| KI-18 | llm | Free-tier LLM: daily quota, 503s, slow tail; `timeout_s` is not a hard total limit | Open |
| KI-19 | rag | Prompt-injection defence lowers the risk but is not a guarantee | Open, by design |
| KI-20 | rag | Citation checking proves the number exists, not that the chunk supports the sentence | Open |
| KI-21 | rag | Retry loop duplicated in `rag.py`; embedder maps a bad API key to `request_rejected` | Open |
| KI-22 | api | Indexing runs in a `BackgroundTask`: lost on restart, no queue, one process | Open, by design |
| KI-23 | api | No authentication or per-user data separation | Open, by design |
| KI-24 | api | Thread-pool starvation, blocked chat during long LLM calls | Open |
| KI-25 | api | Small consistency gaps (ready but vectors missing, same bytes with another name) | Open |
| KI-26 | evaluation | Page-level labels hide chunk-level misses; tiny set; threshold tuned and tested on the same questions | Open |
| KI-27 | frontend | Polling, per-session chat history, UI-side timeout does not stop the server, English widget texts, one browser checked | Open, by design |
| KI-28 | summary / quiz | LLM-verified, not proven: easy distractors, shared blind spots of the verifier, quota cost, no lock on identical requests | Open |
| KI-29 | history | Saved chats are shared (no auth) and keep source excerpts of deleted documents; follow-ups do NOT use history | Open, by design |

## KI-01: Repeated headers/footers are indexed

- **Verified:** in `data/samples/postgresql_performance_guide_vi.pdf`, all 9 pages end with the
  line "Tối ưu hiệu năng PostgreSQL – Tài liệu đào tạo nội bộ Trang N/9", and it lands inside a
  chunk's text. The same boilerplate in every page's last chunk adds noise to its embedding.
- **Not measured:** how much this actually hurts retrieval. No evaluation has been run yet.
- **Decision:** do not fix now. Fix only if the evaluation (Precision@K / Recall@K) shows it
  matters, so the change is justified by numbers.
- **Idea if needed:** drop lines that repeat on most pages. Run the evaluation before and after
  and report both results in `evaluation/report.md`.
- **Evaluation update (2026-11-05):** on 15 answerable questions Hit@5 = 1.000 and MRR = 1.000 with the footers still in
  the chunks, so there is no sign of harm. The before/after experiment above was NOT run, so this does not prove the
  footers are harmless. Keep deferred.

## KI-02: Heading separated from its body

- **Verified:** on the sample, chunk 0 ends with the heading "Mục lục" and its list starts in
  chunk 1. The splitter only looks at length and separators, not document structure.
- **Impact:** a question about a section may retrieve a chunk that has the title but not the answer.

## KI-03: PDF line structure is flattened

- **By design:** inside one PyMuPDF text block, line breaks are turned into spaces (they are
  usually just visual wraps). Bullet lists or table rows that PyMuPDF puts in a single block
  become one run-on line.
- **Not measured** on real documents. The sample's bullet list (chapter 3) came out as separate
  paragraphs because each bullet was its own block.

## KI-04: Partially scanned PDFs are accepted

- A PDF is rejected as scanned only when it has fewer than 25 non-whitespace characters per page
  on average (`MIN_PDF_CHARS_PER_PAGE`). A PDF with some text pages and some image pages passes;
  the image pages are silently missing from the index and only a log warning is emitted.
- Tests cover fully blank and image-only PDFs, **not** the mixed case.

## KI-05: No OCR

Scanned or image-only PDFs are rejected with `ErrorCode.SCANNED_PDF` and a clear message.
Text inside images embedded in an otherwise normal PDF is ignored.

## KI-06: DOCX limits

- No reliable page numbers, so chunks have `page=None` and citations cannot show a page.
- Paragraphs and tables are read. Headers, footers and text boxes are not, and tables nested
  inside table cells are not read separately. (From reading the code; not covered by tests.)

## KI-07: Hyphen re-joining can merge real compounds

A hyphen at a line end followed by a lowercase letter is joined ("con-\ntainer" becomes
"container"). A real compound split across lines, such as "end-\nto-end", becomes "endto-end".
Uppercase and digit continuations are left alone.

## KI-08: Re-indexing can leave stale chunks (resolved)

**Resolved:** `QdrantStore.index_document` deletes every point of the `doc_id` before upserting.
`test_reindexing_with_fewer_chunks_removes_the_stale_ones` covers it and fails if the delete is removed.
Original problem, kept for context:

`doc_id` depends only on the file bytes, and `chunk_id` on `doc_id` and the chunk index. If the
same file is indexed again with a different `CHUNK_SIZE`, `CHUNK_OVERLAP` or cleaning rules, the
new chunks overwrite the old ones with the same index, but if there are now fewer chunks the
extra old ones stay in Qdrant.
**Rule:** delete all points with that `doc_id` before upserting a document. Changing the
embedding model or dimension already requires recreating the collection (see `CLAUDE.md`).

## KI-09: Upload size is checked late (resolved in `main.py`)

**Resolved**, with a correction to the original wording: reading the upload in pieces inside the endpoint is
not enough, because Starlette's multipart parser spools the whole upload to a temporary file BEFORE the
endpoint runs. The real limit therefore lives in `BodySizeLimitMiddleware` (refuses a too-large
`Content-Length` without reading anything; counts bytes of chunked bodies and aborts past the limit),
and `_read_bounded` is the second line. Measured on the real server: a 30 MB upload against the 20 MB
limit got `413` in 0.02 s with 0 bytes sent. Bodies of all other endpoints are capped at 1 MB. Not covered:
the worst case of a chunked body is `limit + one network read` of bytes spooled before the abort.
Original problem, kept for context:

`validate_upload` compares `len(data)` with `MAX_UPLOAD_MB`, so the bytes are already in memory.
The API layer must also cap how much it reads from the request.

## KI-10: Tests depend on a system font

Three PDF tests need a TTF with Vietnamese glyphs (PyMuPDF's built-in fonts have none) and are
skipped if none is found. They ran on this Windows machine (Arial). They have not been run on
Linux or macOS.

## KI-11: `langchain-text-splitters`

- **Decision:** keep it (allowed by `CLAUDE.md`; pinned to 1.1.3).
- **Footprint:** it pulls in `langchain-core` and about 13 other packages (`langsmith`, ...).
- **Upstream bug (verified on 1.1.3):** with regex separators and `keep_separator=False`, the
  regex source text is inserted into chunk text. `chunk_pages` uses `keep_separator=True`, and
  `test_splitter_does_not_leak_regex_source_into_chunks` guards this. **Re-run the tests after
  any upgrade of this package.**

## KI-12: Free-tier embedding throughput (handled in `main.py`)

`POST /documents` now returns `202` immediately and a background job embeds and indexes; the client polls
`GET /documents/{doc_id}` (`processing` -> `ready` | `failed`). The throughput limit itself is unchanged: the
29-chunk sample took about 3 s on a fresh quota, a 500-chunk document would still take minutes. Measured:
the `202` came back in 0.43 s.

- **Measured:** the free tier limits `embed_content` to roughly 100 *texts* per minute (every text in
  a batch counts, not every call). Three consecutive batches of 100 texts took 85 s, 70 s and 78 s,
  including 429 retries. A 500-chunk document would therefore take several minutes.
- **Observed:** the server's `retryDelay` can underestimate: one batch got 429 again after waiting
  22 s and 4 s and only passed on the third retry. `GeminiEmbedder` re-reads the delay every time, and
  the default of 5 retries (`EMBED_MAX_RETRIES`) was enough in every run so far.
- **Consequence:** `embed_documents` is synchronous and blocks for the whole time. The upload endpoint
  in `main.py` must not do it inside the request (use a background task or return a job status).
- There is no proactive rate limiter; we only react to 429.

## KI-13: `index_document` is not atomic

It deletes the old points and then upserts the new ones in batches. If the process dies or Qdrant
fails in between, the document is missing or incomplete until `index_document` is called again
(safe to repeat: it is idempotent). Input is validated before the delete, so bad input never
destroys existing data (tested).

## KI-14: Cover and table-of-contents chunks are retrieved as noise

- **Observed** in `scripts/try_retrieval.py` on the sample: for both unrelated queries the top-1 hit
  was the title chunk or the table-of-contents chunk (score 0.48 to 0.53), and the table-of-contents
  chunk was also hit #3 for a relevant query (0.6746). Generic, topic-less chunks look "close" to
  everything.
- **Not measured:** whether it changes answers. Revisit together with KI-01 once the evaluation exists.

## KI-15: `list_documents` is 1 + N requests

It reads exact counts per `doc_id` with one `facet` call, then one `scroll` per document to get the
filename. Fine for tens of documents; store the filename in SQLite (`db.py`) if the list gets long.

## KI-16: Token-limit pre-check assumption

`gemini-embedding-001` accepts over-long text without error or any "truncated" flag (measured), so
`GeminiEmbedder` calls `count_tokens` for texts longer than 1000 characters and rejects those over
2048 tokens. Texts up to 1000 characters are not counted, which assumes at least half a character
per token (at most 2000 tokens). Typical Vietnamese text is about 4 characters per token (measured on
one repeated phrase only); text full of emoji or rare scripts was not tested.

(KI-14 evaluation update, 2026-11-05: 9 of the 18 prompts sent to the LLM contained a page-1 cover/TOC chunk (q01, q05, q10,
q11, q12, q13, q16, q17, q18). The TOC chunk was top-1 for the two absent-topic questions q17 (sharding, 0.7174) and q18
(MySQL, 0.7123). In q13 it passed the threshold (0.6577) while the chunk with two of the seven answers (0.6437) did not.
Answers stayed faithful; the cost is wasted context and a noisier top-1. Dropping chunks 0 and 1 would NOT separate the
classes (see KI-17).)

(KI-14 addendum, measured with `scripts/try_rag.py`, threshold 0.65: the table-of-contents chunk
scored 0.652 and 0.659 for two answerable questions, so it passed the threshold and was sent to the
LLM as uncited noise. It did no visible harm in 10 questions.)

## KI-17: SCORE_THRESHOLD is provisional and the score ranges overlap

- **Measured** (cosine, top-1 per query, `gemini-embedding-001`, 768 dims, 29-chunk sample):

  | category | n | min | median | max |
  |---|---|---|---|---|
  | answerable | 11 | 0.7515 | 0.7896 | 0.8221 |
  | unrelated (cooking, football) | 2 | 0.4835 | 0.5063 | 0.5292 |
  | related-but-absent (MongoDB sharding / `pg_upgrade` / PostgreSQL sharding) | 3 | 0.5663 | 0.6493 | 0.7038 |

- Unrelated queries are cleanly separated from answerable ones (gap 0.22). Related-but-absent queries
  are **not**: "Sharding trong PostgreSQL" scored 0.7038, only 0.048 below the weakest answerable query,
  and ranks 2-5 of answerable queries (0.65-0.73) sit inside the same band. A single hard threshold
  cannot separate them; with 0.65, `pg_upgrade` fell back by a margin of 0.0007 and PostgreSQL sharding
  went to the LLM, which correctly said the document does not cover it. That is the second safety net.
- The sample is one 9-page document and the groups have 2-3 queries each. **Do not trust 0.65 yet.**
  Re-tune with the evaluation set. Idea to test then: two numbers, a *gate* on the best score
  (answer at all?) and a lower *include* cut-off for which chunks go into the prompt.
- **Evaluation update (2026-11-05, 20 questions, same document, `evaluation/report.md` section 4):** top-1 of the 15
  answerable questions ranges 0.680-0.836; of the 5 that should fall back, 0.481-0.717 (q16 0.6701, q17 0.7174, q18 0.7123,
  q19 0.5427, q20 0.4814). The classes **overlap** by 0.037 (q17 above q11). At 0.65: 0 answerable blocked, 3 of 5 fall-back
  questions pass the gate, and the LLM refused all 3 correctly. The grid optimum 0.68 is a single point and q11 clears it by
  0.0003, so it was NOT adopted. The same topic worded differently moved `pg_upgrade` from 0.6493 to 0.6701, across the
  threshold. Dropping the cover/TOC chunks would not remove the overlap (q17 stays at 0.7144). Measurement only: the
  two-threshold idea showed no page-level gain (recall 0.978 unchanged, more chunks, lower precision); a chunk-level label set
  is needed to judge it (q13 suggests a gain there).

## KI-18: Free-tier LLM behaviour

- **Daily quota:** `gemini-3.5-flash` hit `GenerateRequestsPerDayPerProjectPerModel-FreeTier` after about 45
  calls in one day (429 with retryDelay ~24000 s). The pipeline fails fast with `RATE_LIMITED` instead of
  sleeping for hours. Budget the evaluation run (20+ questions, plus any LLM judge) against this.
- **Measured latency** (5 real chunks, ~1440 prompt tokens, single question, 2 benchmark rounds with
  retries disabled, `scripts/bench_llm.py`): `gemini-3.5-flash` default thinking median 4.1 s over 9 runs
  (3.0-19.0 s, ~240-430 thinking tokens); `minimal` median 1.7 s over 8 runs (1.1-5.6 s, no thinking
  tokens); `gemini-3.1-flash-lite` `minimal` median 1.6 s over 7 runs but 2 of 7 took 13-18 s; its default
  setting failed with 503/504 in 5 of 9 calls and once took 48.8 s. Server load, not prompt size, explains
  the tail (answers had 40-60 output tokens).
- **Timeout is not a hard limit:** the SDK passes `timeout` to httpx as a number (applied per phase, not to
  the whole call) and sends `X-Server-Timeout`. In `try_rag.py` one question took 87 s: a first attempt of
  roughly 84 s ended in 504 DEADLINE_EXCEEDED, then the retry succeeded in ~1.3 s. Cause not investigated
  further. A real wall-clock cap would need a thread/async wrapper.
- **Evaluation update (2026-11-05, `gemini-3.1-flash-lite`, `minimal`, 18 LLM calls):** median `llm_ms` 1.23 s, but 2 of 18 took
  longer than 10 s: q04 61.8 s (one retry after `HTTP 500`; the first attempt ran about the 60 s `timeout_s`, inferred) and q09
  13.2 s with no retry. `gemini-3.5-flash` was out of daily quota during that run (429 asking to wait 14991 s), so its
  answer quality was not measured.
- **Summary / quiz update (2026-11-05):** a real quiz of 10 questions with verification took 72 s: the server log shows one
  `HTTP 504 DEADLINE_EXCEEDED` (the first attempt ran about the 60 s timeout) and one retry. The UI therefore waits 120 s for
  summary and quiz (chat stays at 60 s).
- 504 is retried like 429/5xx by design (same policy as embeddings); with 5 retries the worst case is
  minutes. Consider `EMBED_MAX_RETRIES` lower for interactive use or a separate LLM retry setting.

## KI-19: Prompt injection

Retrieved text is wrapped in markers with a random per-request token and the system prompt says to treat
it as data (tested: a chunk cannot forge the end marker). This is not a guarantee: a model can still be
swayed by text inside the data block, and a chunk can imitate a chunk header such as "[2] (x.pdf, trang 1)"
to confuse numbering. **Not tested against the real model** with a malicious document; only the prompt
construction is tested.

## KI-20: Citation checking is shallow

`resolve_citations` removes numbers that do not exist and flags which sources were cited. It does not check
that the cited chunk really supports the sentence, and a regex cannot tell `[1]` as a citation from `[1]` in
code the model quotes (numbers of 4+ digits, such as years, are ignored). An answer with no citation at all
is returned as it is (warning in the log). `used_fallback` only means "the LLM was not called": the LLM may
still answer "the documents do not cover this" for chunks that passed the threshold.

## KI-21: Duplicated retry loop, bad-key mapping

- `GeminiLLM._call_with_retry` repeats the structure of `GeminiEmbedder._call_with_retry` (and imports the
  private `_retry_delay_from`). Extract a shared helper if a third caller appears.
- **Measured:** a wrong API key returns HTTP 400 INVALID_ARGUMENT "API key not valid", not 401/403.
  `GeminiLLM` maps it to `auth_error` (message check); `GeminiEmbedder` still reports `request_rejected`.

## KI-22: Background indexing is not durable

- `BackgroundTasks` run inside the server process. If it stops, the work is lost. On the next start every
  `processing` row is marked `failed` with code `interrupted` (verified on the real server) and the user
  must upload the file again; re-uploading is safe (the old vectors are replaced).
- No queue, no concurrency limit, no progress percentage: only `processing`. Several large uploads at once
  all compete for the same embedding quota, and each job occupies one worker thread for its whole duration.
- Single process only: `claim_document` is atomic across threads and processes (SQLite `BEGIN IMMEDIATE`), but
  the uploaded file bytes and chunks are handed to the job in memory, so a second process cannot continue it.
- A document that was `failed` after a partial upsert keeps its partial vectors until it is uploaded again or
  deleted. Chat never reads them because it only searches `ready` documents.

## KI-23: No authentication

Anyone who can reach the API can upload, list, delete and ask. All documents are shared by all callers. CORS
allows only the local Streamlit origin, but CORS is a browser rule, not access control. Fine for a local demo,
not for a deployment.

## KI-24: Thread pool and long requests

All endpoints are plain `def` (run in AnyIO's pool of 40 threads by default). A chat call holds a thread for
the whole LLM call, including retries (up to minutes, KI-18), and every running indexing job holds one too.
Enough slow requests at once can use up the pool, and then even `/health` waits. Not reproduced, from reading
the design. Mitigations if needed: lower retries for chat, a semaphore for indexing jobs.

## KI-25: Small consistency gaps

- If a document is `ready` in SQLite but its vectors were removed from Qdrant by hand (volume wiped), a re-upload
  returns `ready` and does not re-index. Delete the document, then upload again.
- Uploading the same bytes under a different name returns the original record and keeps the original file name.
- The multi-step delete (vectors, then row, then file) is not atomic. Vectors go first on purpose: if Qdrant
  is down nothing changed and the request can be retried. A failure to remove the stored file only logs a warning.
- `/docs` (Swagger UI) returns HTML from the server, but its scripts and styles load from a CDN, so a full
  render needs internet. Only the HTML and `/openapi.json` were checked, not a browser render.
- A request body that is not valid UTF-8 (some Windows shells send cp1252 for Vietnamese text via curl) gets
  `400 bad_request`; found on the real server and covered by a test.

## KI-26: Limits of the evaluation itself

- **Page-level labels hide chunk-level misses.** q13 needs three chunks of page 4 but only two reached the prompt (the third scored
  0.6437 < 0.65); Recall@5 at page level is still 1.0. Label the evidence chunks to see this.
- **Small and single-document:** 20 questions (15 answerable), one 9-page PDF. One question moves a rate by 5-7 points; there are no
  meaningful confidence intervals.
- **The threshold is judged on the questions it was tuned with.** Four questions (q05, q16, q17, q19) also served to pick 0.65.
  Use a dev set for tuning and a separate test set for reporting.
- **The LLM judge is imperfect and here shared the model with the answerer** (`gemini-3.1-flash-lite`). A second opinion (Claude)
  agreed on all 18 verdicts and flagged one borderline (q11); two LLMs agreeing is not ground truth. A human should re-read the
  5 random verdicts printed by the script and the borderline case.
- **Not measured:** `gemini-3.5-flash` answers (daily quota), documents longer than 9 pages, DOCX/TXT (no page numbers), more than
  one document, the effect of the repeated footers (KI-01) in isolation.
- The cached run in `evaluation/cache/` is what makes the numbers reproducible; without it, LLM answers can differ slightly
  (temperature 0.2).

## KI-27: Limits of the Streamlit client (`frontend/`)

- **Polling, not push.** While a document is `processing` the sidebar fragment calls `GET /documents` every 2 s; it gives up
  after 180 s (the user can press "Kiểm tra lại"; after 30 s a "free tier is slow" hint is shown). Measured on the sample
  PDF: `processing` -> `ready` in about 3 s, seen by the page without any click. A 500-chunk document (KI-12) would exceed
  the 180 s limit and need the manual re-check. Every interaction also costs one `GET /health` and one `GET /documents`.
- **The 60 s UI-side timeout only ends the UI's wait** (`requests` read timeout, per socket read, not a hard wall-clock cap).
  The server keeps the LLM call running, so the thread stays busy (KI-18, KI-24) and the answer is lost. The message says the
  request may still be running. Checked with a real socket that never answers (timeout fired at the configured time).
- **A browser interaction during a chat call cuts the run.** The question is marked `sent` before the call, so it is never sent
  twice; the user sees "bị gián đoạn" and has to ask again. Not reproduced in a real browser (needs a click inside the 1-60 s
  of a call); covered by an AppTest that sets the state directly.
- **Chat history lives in `st.session_state` only**: a browser reload or a new tab starts an empty conversation (documents stay,
  they are on the server). No persistence of conversations although `db.py` has the tables.
- **No authentication, single shared document set** (KI-23). `.streamlit/config.toml` makes Streamlit listen on 127.0.0.1 only
  (its default is every network interface); that is not access control.
- **Upload size** is capped in the widget from `UPLOAD_MAX_MB` (default 20) which must be kept equal to the backend's
  `MAX_UPLOAD_MB`; the backend stays the real limit (a 30 MB upload got `413` in 0.05 s through the client).
- **Some texts are Streamlit's own and stay English** ("Upload", "20MB per file", "Deploy"). Not localised.
- Browser check used Playwright with the installed Microsoft Edge (installed in a scratch venv outside the project, not a
  project dependency); no other browser or mobile layout was looked at.
- The client trusts the Vietnamese message in the backend's error body. If a proxy answers with another body, the client shows
  its own generic text by status code instead of the raw body.

## KI-28: Summary and quiz generation (`app/summary.py`, `app/quiz.py`)

**Cost per request** (each LLM call spends free-tier quota; results are stored in SQLite table `generations`, so a repeated
request costs 0 calls, measured: 0.0-0.1 s):

| Request | LLM calls |
|---|---|
| Summary, document fits 60 000 characters (the 9-page sample is 17 640) | 1 |
| Summary, longer | one per group of at most 40 000 characters + 1 merge = at most 7; beyond 6 groups the end is left out and the response says so (`truncated`, `warning`) |
| Quiz without verification | 1 |
| Quiz with verification (default) | 2 |

Measured with `gemini-3.1-flash-lite` + `minimal` (3.5-flash was out of quota): short summary 1.6 s, detailed 3.8 s, quiz of 5 with
verification 6.8 s, map-reduce with small limits (4 groups) 5 calls in 9.0 s, a capped one (2 groups, 14 of 29 chunks) 3 calls.

**What the code checks, and what it does not**

- Checked in code, not by trusting the model: 4 distinct options, valid key, non-empty explanation, a quote of at least 15
  characters that occurs verbatim (after NFC + whitespace normalisation, case-sensitive) in the chunk the question cites; the page
  always comes from that chunk. In the real runs this dropped 1 of 11 questions of a 10-question request (`quote_not_in_source`).
- The position of the correct option is assigned by code in balanced blocks of four, so a model that always answers "A" cannot
  pass that bias on (tested over hundreds of seeds).
- A quote proves the answer is in the document, **not that the question is good**. Read by hand against the PDF (15 distinct real
  questions): none wrong, none with two clearly correct options, one mildly ambiguous (an option that is a downstream symptom
  described elsewhere in the document), and many distractors are unrelated concepts, so the quizzes are easy. The model also
  tends to ask twice about the same chunk even though the sources are spread over the document.
- **The verifier is the same model family and sees only the cited chunk.** On the real quizzes it accepted 20 of 20 questions; in a
  negative control (4 questions, 3 deliberately wrong) it rejected 3 of 3 and accepted the correct one. That shows it can say no,
  with n = 4; it did not catch the mildly ambiguous question above, and it cannot notice that another chunk contradicts the key.
  A quiz whose verification could not run (quota, unreadable reply) is returned flagged "unverified" and is not stored.
- Page-range quizzes use the chunk's page, so DOCX/TXT chunks (no pages) are never selected when a range is given. Topic quizzes pick
  chunks by embedding search (12 nearest), so a vague topic can give few distinct chunks (a topic + page-8 quiz used 2 chunks).

**Other limits**

- Budgets are characters, not tokens (no `count_tokens` call). Consecutive chunks of a page overlap by `CHUNK_OVERLAP`, so the
  budget counts a little duplicated text.
- Two identical requests that arrive at the same moment both call the model (no lock); the second stored result replaces the first.
- The stored result is keyed by the indexed text, model + thinking level and the prompt texts; changing any prompt invalidates it
  automatically. Old rows of other keys stay until the document is deleted (they cascade with it).
- The API returns the answer key and explanation with the questions: grading happens in the Streamlit session, there is no
  server-side exam mode. Fine for study use; not for assessment.
- Real runs covered only the 9-page sample PDF (no DOCX/TXT, no document near the 60 000-character budget; the map-reduce path was
  run with artificially small limits).

## KI-29: Conversation history (`conversations` / `messages`, `GET|DELETE /conversations`)

**What it is.** Every chat exchange is saved by the backend while it answers: the question, the answer, the sources (JSON), `used_fallback`,
`best_score`, the latency and `truncated`, in the conversation given by `conversation_id` (omitted = a new conversation titled with the first
question). Users can list, open and delete saved conversations. Opening one only reads SQLite: **no LLM call** (measured on the real stack:
0 `POST /chat` caused by opening; 1.3 s to open; the list and the messages survived a restart of Streamlit and of the API).

**What it is NOT: follow-up understanding.** Earlier messages are *not* sent to the model and are not used to rewrite or interpret the next
question ("what about it?" is answered as an isolated question). `conversation_id` only says where to save. The UI says so under the chat
title and the OpenAPI description of `conversation_id` says so. Rewriting a follow-up with the history needs an extra LLM call per question
and is out of scope.

**Saving is all-or-nothing, and never costs the answer.** `Database.save_exchange` writes the conversation row (when new), the user message,
the assistant message and the `updated_at` bump in one `BEGIN IMMEDIATE` transaction (tested with a failing trigger: nothing is left behind,
no conversation without messages, no question without its answer; and with 8 simultaneous exchanges into one conversation). If saving fails
for any reason the answer (which cost an LLM call) is still returned with `saved: false`, the error is logged, and the UI shows a warning.
A wrong `conversation_id` is a `404` *before* the model is called, so it costs no quota. A failed LLM call saves nothing.

**Privacy and sharing (the important limits).**
- There is no authentication (KI-23): **every client that can reach the API sees, opens and deletes every conversation**, and can write into any
  conversation whose 32-character id it knows. Ids are random but they are not secrets.
- Questions, answers and 200-character excerpts of the source passages are stored in plain text in `data/app.sqlite3` and kept until the
  conversation is deleted (no expiry, no encryption).
- **Deleting a document does not delete the history that mentions it**: the conversation still loads and its sources are marked
  "Tài liệu đã bị xóa" (`document_exists: false`, listed in `missing_doc_ids`), but the stored excerpts of that document remain readable.
  To remove them, delete the conversations. There is deliberately no foreign key from a conversation to `documents`.

**Other limits.**
- No rename (`PATCH`), no search, no export, no "delete all", no retention policy. The sidebar lists at most the 100 most recently active
  conversations (the API page limit); the title is the first question and never changes.
- "Xóa hội thoại" clears the screen and detaches from the saved conversation (the next question starts a new one); it never deletes anything
  saved. It overlaps with "Cuộc trò chuyện mới" on purpose: one is a clean-up of the screen, the other the explicit way to start again.
- Two browser tabs on the same conversation append to it independently; a tab does not update live.
- A request cut off by a page redraw may still have been saved by the server: it then appears in the history even though the UI showed
  "bị gián đoạn".
- The schema gained columns (`doc_ids_json`, `updated_at`, and the answer fields of `messages`). `init_db` adds them to an existing database
  (verified on the real, old-schema `app.sqlite3` and in a test); there is no down-migration.
