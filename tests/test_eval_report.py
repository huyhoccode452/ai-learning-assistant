"""Tests for evaluation/report.py with small synthetic results (no files, no network). The retrieval
numbers below are worked out by hand in the comments."""

import copy

from evaluation.report import find_missing, flag_questions, render_report, summarize

CHUNKS_PER_PAGE = {"1": 2, "2": 1, "3": 2, "4": 1}
LATENCY = {"embed_ms": 100.0, "search_ms": 10.0, "llm_ms": 900.0, "total_ms": 1010.0}


def hit(score: float, page: int, chunk: int) -> dict:
    return {"score": score, "page": page, "chunk_index": chunk}


def answered(answer: str, pages: list[int], total_ms: float = 1010.0) -> dict:
    return {"answer": answer, "used_fallback": False, "best_score": 0.8, "retries": 0, "truncated": False,
            "invalid_citations": [], "latency": {**LATENCY, "total_ms": total_ms},
            "sources": [{"number": i, "page": p, "chunk_index": i, "score": 0.8, "cited": i == 1} for i, p in enumerate(pages, 1)]}


def verdict(correctness: str, refusal: bool = False, claims: list[str] | None = None) -> dict:
    return {"correctness": correctness, "is_refusal": refusal, "faithful": not claims, "unsupported_claims": claims or [],
            "notes": "ghi chú", "from_cache": False}


def question(qid: str, type_: str, pages: list[int], hits: list[dict], e2e=None, judge=None, outcome=None) -> dict:
    return {"id": qid, "type": type_, "question": f"Câu {qid}?", "reference_answer": f"Đáp án {qid}", "relevant_pages": pages,
            "expect_fallback": not pages, "hits": hits, "retrieval_error": None, "e2e": e2e, "e2e_error": None,
            "judge": judge, "judge_error": None, "outcome": outcome}


def results() -> dict:
    return {
        "meta": {"date": "2026-01-01", "document": "d.pdf", "doc_id": "abc", "num_pages": 4, "num_chunks": 6, "chunk_size": 800,
                 "chunk_overlap": 120, "top_k": 5, "ks": [3, 5], "score_threshold": 0.7, "embedding_model": "emb",
                 "embedding_dim": 768, "llm_model": "llm", "llm_thinking": "minimal", "env_llm_model": "llm",
                 "env_llm_thinking": "minimal", "judge_model": "llm", "judge_thinking": "low", "seed": 1,
                 "sample_ids": ["a1"], "stages": ["retrieval", "end_to_end", "judge"]},
        "chunks_per_page": CHUNKS_PER_PAGE,
        "questions": [
            # a1: pages [2]; retrieved pages 2, 1, 3 -> P@3 1/3, R 1, hit 1, RR 1
            question("a1", "single", [2], [hit(0.80, 2, 2), hit(0.70, 1, 0), hit(0.60, 3, 3)],
                     answered("Đáp án a1 [1]", [2]), verdict("correct"), "correct"),
            # a2: pages [3, 4]; retrieved 3, 3, 1, 4, 2 -> P@3 2/3, R@3 1/2, R@5 1, RR 1
            question("a2", "multi", [3, 4], [hit(0.75, 3, 3), hit(0.74, 3, 4), hit(0.70, 1, 1), hit(0.69, 4, 5), hit(0.60, 2, 2)],
                     answered("Một phần [1]", [3, 3, 1]), verdict("partially_correct"), "partially_correct"),
            # f1 is answered by the LLM (refusal), f2 falls back
            question("f1", "insufficient", [], [hit(0.72, 1, 1)], answered("Tài liệu không có.", [1]), verdict("correct", True), "correct_refusal"),
            question("f2", "unrelated", [], [hit(0.50, 1, 0)],
                     {"answer": "fallback", "used_fallback": True, "best_score": 0.5, "retries": 0, "truncated": False,
                      "invalid_citations": [], "sources": [], "latency": {**LATENCY, "llm_ms": 0.0, "total_ms": 300.0}},
                     None, "correct_fallback"),
        ],
    }


REVIEW = {"reviewer": "R", "note": "n", "verdicts": {
    "a1": {"correctness": "correct", "faithful": True, "comment": "ok"},
    "a2": {"correctness": "incorrect", "faithful": True, "comment": "tôi thấy sai", "borderline": True}}}


def test_summary_numbers_match_the_hand_computation() -> None:
    s = summarize(results())
    # P@3: a1 1/3, a2 2/3 -> 0.5 ; R@3: 1 and 1/2 -> 0.75 ; R@5: 1 and 1 -> 1 ; MRR: 1 and 1 -> 1
    assert round(s["overall"].precision[3], 6) == 0.5 and round(s["overall"].recall[3], 6) == 0.75
    assert s["overall"].recall[5] == 1.0 and s["overall"].mrr == 1.0 and s["overall"].hit[3] == 1.0
    # top-1: answerable 0.80 and 0.75 ; should-fall-back 0.72 and 0.50 -> gap 0.75 - 0.72 = 0.03, no overlap
    assert not s["separation"].overlaps and round(s["separation"].gap, 6) == 0.03
    # latency: totals 1010, 1010, 1010, 300 ms -> median 1010 ; only 3 of 4 called the LLM
    assert s["latency"]["median"] == 1010.0 and s["latency"]["llm_n"] == 3 and s["latency"]["fallback_median"] == 300.0
    assert set(s["by_type"]) == {"single", "multi"}


def test_the_report_has_the_expected_numbers_and_sections() -> None:
    text = render_report(results(), "Phân tích tay ở đây.", REVIEW)
    for fragment in ("## 3. Truy xuất", "## 4. Nghiên cứu ngưỡng", "## 5. End-to-end", "## 6. Độ trễ", "Phân tích tay ở đây.",
                     "| Precision@K | 0.500 | ", "tách được", "LLM từ chối đúng", "Fallback đúng", "Đúng một phần",
                     "Ý kiến thứ hai: R", "không đồng ý", "ranh giới", "evaluate.py --report-only",
                     "chỉ 'Đúng', câu bị chặn tính là sai): 1/2", "Đúng hoặc đúng một phần: 2/2", "Xử lý đúng: 2/2"):
        assert fragment in text, fragment
    assert "Dữ liệu chưa đầy đủ" not in text


def test_the_report_is_honest_about_missing_data() -> None:
    data = results()
    data["questions"][1]["e2e"] = None  # a2 never got an answer (quota)
    data["questions"][1]["judge"] = None
    data["questions"][1]["outcome"] = None
    assert find_missing(data) == {"retrieval": [], "end_to_end": ["a2"], "judge": []}
    text = render_report(data, None, None)
    assert "Dữ liệu chưa đầy đủ" in text and "end_to_end` thiếu: a2" in text
    assert "THIẾU" in text and "Chưa có (`evaluation/analysis_vi.md` chưa được viết)" in text


def test_an_answer_without_a_verdict_is_reported_as_judge_missing() -> None:
    data = results()
    data["questions"][0]["judge"] = None
    assert find_missing(data)["judge"] == ["a1"]  # f2 is a fallback and needs no verdict


def test_skipping_the_llm_stages_is_not_reported_as_missing() -> None:
    data = results()
    data["meta"]["stages"] = ["retrieval"]
    for q in data["questions"]:
        q["e2e"] = q["judge"] = q["outcome"] = None
    assert find_missing(data) == {"retrieval": [], "end_to_end": [], "judge": []}
    text = render_report(data, None, None)
    assert "không được chạy (`--skip-llm`)" in text and "## 3. Truy xuất" in text


def test_a_report_without_any_retrieval_data_does_not_crash() -> None:
    data = results()
    data["meta"]["stages"] = []
    for q in data["questions"]:
        q["hits"] = q["e2e"] = q["judge"] = q["outcome"] = None
    text = render_report(data, None, None)
    assert "Không có dữ liệu truy xuất" in text and find_missing(data)["retrieval"] == ["a1", "a2", "f1", "f2"]


def test_overlapping_classes_are_called_out() -> None:
    data = copy.deepcopy(results())
    data["questions"][2]["hits"] = [hit(0.77, 1, 1)]  # a should-fall-back question now scores above a2 (0.75)
    text = render_report(data, None, None)
    assert "CHỒNG LẤN" in text and "tách được" not in text.split("### 4.1")[0].split("## 4.")[1]


def test_flags_name_the_reason() -> None:
    data = results()
    data["questions"][0]["outcome"] = "wrongly_blocked"
    data["questions"][0]["e2e"] = {**data["questions"][0]["e2e"], "used_fallback": True}
    data["questions"][3]["e2e"]["used_fallback"] = False  # f2 "passed" the threshold although it should fall back
    data["questions"][3]["outcome"] = "wrongly_answered"
    data["questions"][3]["judge"] = verdict("incorrect", claims=["bịa"])
    flags = dict(flag_questions(data, summarize(data)))
    assert any("chặn nhầm" in why for why in flags["a1"])
    assert any("vượt ngưỡng" in why for why in flags["f2"]) and any("không trung thực" in why for why in flags["f2"])
    assert "a2" in flags and any("Recall@5" not in why for why in flags["a2"])  # a2 is partially correct
