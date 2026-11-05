# Known issues

Issues we are aware of and have deliberately not fixed yet. Each entry says what we
verified, what we did not, and when to revisit. Update the status when one is resolved.

Last updated after `app/prompts.py` and `app/rag.py` (192 tests + 3 integration tests).

| ID | Area | Issue | Status |
|----|------|-------|--------|
| KI-01 | ingestion | Repeated page headers/footers are indexed | Deferred until evaluation data exists |
| KI-02 | ingestion | A heading can end a chunk, away from its body | Open |
| KI-03 | ingestion | PDF line structure (lists, tables) is flattened | Open, by design |
| KI-04 | ingestion | Partially scanned PDFs are accepted | Open |
| KI-05 | ingestion | No OCR for scanned PDFs | Out of scope |
| KI-06 | ingestion | DOCX: no page numbers, some content not read | Open |
| KI-07 | ingestion | Line-end hyphen re-joining can merge real compounds | Open |
| KI-08 | vectorstore | Re-indexing can leave stale chunks | Resolved in `QdrantStore.index_document` |
| KI-09 | api | Upload size limit is checked after the file is in memory | Must handle in `main.py` |
| KI-10 | tests | Some tests depend on a system font | Open |
| KI-11 | dependencies | `langchain-text-splitters` footprint and a known upstream bug | Decided: keep |
| KI-12 | embeddings | Free-tier throughput is ~100 texts per 70-85 s; embedding blocks | Open, handle in `main.py` |
| KI-13 | vectorstore | `index_document` is delete-then-upsert, not atomic | Open, by design |
| KI-14 | retrieval | Cover/TOC chunks are retrieved as noise | Open, revisit with evaluation data |
| KI-15 | vectorstore | `list_documents` makes one extra request per document | Open |
| KI-16 | embeddings | Token-limit pre-check rests on an assumption for short texts | Open |
| KI-17 | rag | `SCORE_THRESHOLD` is provisional; related-but-absent queries overlap the answerable range | Open, re-tune with the evaluation set |
| KI-18 | llm | Free-tier LLM: daily quota, 503s, slow tail; `timeout_s` is not a hard total limit | Open |
| KI-19 | rag | Prompt-injection defence lowers the risk but is not a guarantee | Open, by design |
| KI-20 | rag | Citation checking proves the number exists, not that the chunk supports the sentence | Open |
| KI-21 | rag | Retry loop duplicated in `rag.py`; embedder maps a bad API key to `request_rejected` | Open |

## KI-01: Repeated headers/footers are indexed

- **Verified:** in `data/samples/postgresql_performance_guide_vi.pdf`, all 9 pages end with the
  line "Tối ưu hiệu năng PostgreSQL – Tài liệu đào tạo nội bộ Trang N/9", and it lands inside a
  chunk's text. The same boilerplate in every page's last chunk adds noise to its embedding.
- **Not measured:** how much this actually hurts retrieval. No evaluation has been run yet.
- **Decision:** do not fix now. Fix only if the evaluation (Precision@K / Recall@K) shows it
  matters, so the change is justified by numbers.
- **Idea if needed:** drop lines that repeat on most pages. Run the evaluation before and after
  and report both results in `evaluation/report.md`.

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

## KI-09: Upload size is checked late (action for `main.py`)

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

## KI-12: Free-tier embedding throughput

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
