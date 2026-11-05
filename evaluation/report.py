"""Turns evaluation/results.json into evaluation/report.md (Vietnamese).

Every number in the report is computed here from the raw per-question records of a real run; nothing
is typed in by hand. Text that needs a human (root causes, improvement ideas) lives in
evaluation/analysis_vi.md and is inserted verbatim. Missing data (a quota that ran out, --skip-llm)
is reported as missing, never replaced by a guess.
"""

from collections import Counter
from typing import Any

from evaluation.metrics import (
    RetrievalScores,
    ScoredHit,
    average_scores,
    class_separation,
    frange,
    median,
    percentile,
    recommend_threshold,
    score_retrieval,
    threshold_study,
    two_threshold_row,
)

THRESHOLD_GRID = frange(0.55, 0.80, 0.01)
INCLUDE_OFFSETS = (0.0, 0.02, 0.05, 0.10)
SLOW_SECONDS = 10.0  # a question slower than this (or 3x the median) is listed as doubtful

OUTCOME_VI = {
    "correct": "Đúng",
    "partially_correct": "Đúng một phần",
    "incorrect": "Sai",
    "wrongly_blocked": "Bị chặn nhầm",
    "correct_fallback": "Fallback đúng",
    "correct_refusal": "LLM từ chối đúng",
    "wrongly_answered": "Trả lời nhầm",
    "unjudged": "Chưa chấm",
}
TYPE_VI = {"single": "single (1 chunk)", "multi": "multi (nhiều chunk)",
           "insufficient": "insufficient (thiếu bằng chứng)", "unrelated": "unrelated (không liên quan)"}


# ---------------------------------------------------------------------- formatting


def f3(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def secs(ms: float | None) -> str:
    return "–" if ms is None else f"{ms / 1000:.1f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pages_text(pages: list[int | None]) -> str:
    return ", ".join("?" if p is None else str(p) for p in pages) or "–"


# ---------------------------------------------------------------------- analysis of the raw records


def find_missing(results: dict) -> dict[str, list[str]]:
    """Questions a stage could not finish (quota, errors). Stages that were not requested are not 'missing'."""
    stages = results["meta"]["stages"]
    questions = results["questions"]
    return {
        "retrieval": [q["id"] for q in questions if q["hits"] is None],
        "end_to_end": [q["id"] for q in questions if "end_to_end" in stages and q["e2e"] is None],
        "judge": [q["id"] for q in questions
                  if "judge" in stages and q["e2e"] and not q["e2e"]["used_fallback"] and q["judge"] is None],
    }


def _corpus_pages(results: dict) -> list[int | None]:
    pages: list[int | None] = []
    for key, count in results["chunks_per_page"].items():
        pages += [None if key == "None" else int(key)] * count
    return pages


def _top1(question: dict) -> float | None:
    return max((h["score"] for h in question["hits"]), default=None) if question["hits"] is not None else None


def summarize(results: dict) -> dict[str, Any]:
    meta, questions = results["meta"], results["questions"]
    ks = tuple(meta["ks"])
    corpus = _corpus_pages(results)

    scores: dict[str, RetrievalScores] = {}
    for q in questions:
        if q["hits"] is not None and not q["expect_fallback"]:
            scores[q["id"]] = score_retrieval([h["page"] for h in q["hits"]], q["relevant_pages"], ks, corpus)
    by_type = {
        t: average_scores([scores[q["id"]] for q in questions if q["type"] == t and q["id"] in scores])
        for t in ("single", "multi")
        if any(q["type"] == t and q["id"] in scores for q in questions)
    }

    top1 = {q["id"]: _top1(q) for q in questions if _top1(q) is not None}
    answerable = [q for q in questions if not q["expect_fallback"] and q["id"] in top1]
    fallback = [q for q in questions if q["expect_fallback"] and q["id"] in top1]
    study = separation = recommendation = None
    if answerable and fallback:
        a_scores, f_scores = [top1[q["id"]] for q in answerable], [top1[q["id"]] for q in fallback]
        study = threshold_study(a_scores, f_scores, THRESHOLD_GRID)
        separation = class_separation(a_scores, f_scores)
        recommendation = recommend_threshold(study)

    two: list = []
    if answerable and fallback:
        pairs = [([ScoredHit(h["score"], h["page"]) for h in q["hits"]], q["relevant_pages"]) for q in answerable]
        falls = [[ScoredHit(h["score"], h["page"]) for h in q["hits"]] for q in fallback]
        gates = sorted({meta["score_threshold"], recommendation.threshold} if recommendation else {meta["score_threshold"]})
        for gate in gates:
            for offset in INCLUDE_OFFSETS:
                two.append(two_threshold_row(pairs, falls, gate, round(gate - offset, 6)))

    done = [q for q in questions if q["e2e"]]
    called = [q for q in done if not q["e2e"]["used_fallback"]]
    totals = [q["e2e"]["latency"]["total_ms"] for q in done]
    latency = None
    if totals:
        latency = {
            "n": len(totals), "median": median(totals), "p95": percentile(totals, 95), "max": max(totals),
            "slowest": max(done, key=lambda q: q["e2e"]["latency"]["total_ms"])["id"],
            "llm_n": len(called),
            "llm_median": median([q["e2e"]["latency"]["total_ms"] for q in called]) if called else None,
            "llm_p95": percentile([q["e2e"]["latency"]["total_ms"] for q in called], 95) if called else None,
            "llm_max": max((q["e2e"]["latency"]["total_ms"] for q in called), default=None),
            "fallback_median": median([q["e2e"]["latency"]["total_ms"] for q in done if q["e2e"]["used_fallback"]])
            if len(called) < len(done) else None,
            "retried": sum(1 for q in done if q["e2e"]["retries"] > 0),
            "retries": sum(q["e2e"]["retries"] for q in done),
        }
    margin = None
    if recommendation:
        nearest_id, nearest = min(top1.items(), key=lambda item: abs(item[1] - recommendation.threshold))
        margin = (nearest_id, nearest, abs(nearest - recommendation.threshold))
    return {
        "margin": margin,
        "scores": scores, "overall": average_scores(list(scores.values())) if scores else None,
        "by_type": by_type, "top1": top1, "study": study, "separation": separation,
        "recommendation": recommendation, "two": two, "latency": latency,
        "outcomes": Counter(q["outcome"] for q in questions if q["outcome"]),
        "answerable": answerable, "fallback": fallback, "ks": ks,
    }


def flag_questions(results: dict, summary: dict) -> list[tuple[str, list[str]]]:
    """Failed or doubtful questions with the measured reason(s). Root causes are written by hand."""
    threshold = results["meta"]["score_threshold"]
    flags: list[tuple[str, list[str]]] = []
    median_total = summary["latency"]["median"] if summary["latency"] else None
    for q in results["questions"]:
        why: list[str] = []
        scores = summary["scores"].get(q["id"])
        if scores:
            kmax = max(summary["ks"])
            if scores.hit[kmax] == 0:
                why.append(f"không truy xuất được trang đúng trong top-{kmax}")
            elif scores.mrr < 1:
                why.append(f"trang đúng không đứng đầu (hạng đầu tiên = {round(1 / scores.mrr)})")
            if scores.recall[kmax] < 1 and len(q["relevant_pages"]) > 1:
                found = round(scores.recall[kmax] * len(q["relevant_pages"]))
                why.append(f"Recall@{kmax} = {found}/{len(q['relevant_pages'])} trang đúng")
        top1 = summary["top1"].get(q["id"])
        outcome, e2e, judge = q["outcome"], q["e2e"], q["judge"]
        if outcome == "wrongly_blocked":
            why.append(f"bị ngưỡng {threshold} chặn nhầm (top-1 = {top1:.4f})")
        if outcome in ("partially_correct", "incorrect", "wrongly_answered") and judge:
            why.append(f"judge: {judge['correctness']} — {judge['notes']}")
        if judge and not judge["faithful"]:
            why.append("không trung thực: " + "; ".join(judge["unsupported_claims"]))
        if q["expect_fallback"] and e2e and not e2e["used_fallback"]:
            why.append(f"vượt ngưỡng dù tài liệu không có đáp án (top-1 = {top1:.4f}); kết quả: "
                       f"{OUTCOME_VI.get(outcome, outcome)}")
        if e2e:
            total_s = e2e["latency"]["total_ms"] / 1000
            if total_s > SLOW_SECONDS or (median_total and e2e["latency"]["total_ms"] > 3 * median_total):
                why.append(f"chậm: {total_s:.1f} s (retry: {e2e['retries']})")
            elif e2e["retries"]:
                why.append(f"có {e2e['retries']} lần retry")
            if e2e["truncated"] or e2e["invalid_citations"]:
                why.append(f"truncated={e2e['truncated']}, trích dẫn không tồn tại={e2e['invalid_citations']}")
        if q["e2e_error"] or q["judge_error"] or q["retrieval_error"]:
            why.append(f"lỗi: {q['retrieval_error'] or q['e2e_error'] or q['judge_error']}")
        if why:
            flags.append((q["id"], why))
    return flags


# ---------------------------------------------------------------------- sections


def _setup(results: dict) -> str:
    m = results["meta"]
    rows = [
        ["Ngày chạy (UTC)", m["date"]],
        ["Tài liệu", f"`{m['document']}` — {m['num_pages']} trang, {m['num_chunks']} chunk, doc_id `{m['doc_id']}`"],
        ["Số câu hỏi", str(len(results["questions"]))],
        ["Mô hình embedding", f"`{m['embedding_model']}`, {m['embedding_dim']} chiều"],
        ["Mô hình trả lời", f"`{m['llm_model']}`, thinking = `{m['llm_thinking'] or 'mặc định'}`"],
        ["Mô hình chấm (judge)", f"`{m['judge_model']}`, thinking = `{m['judge_thinking']}`, temperature = 0"],
        ["Chunking", f"chunk_size = {m['chunk_size']}, chunk_overlap = {m['chunk_overlap']} (ký tự)"],
        ["Truy xuất", f"top_k = {m['top_k']}; K đánh giá = {', '.join(map(str, m['ks']))}"],
        ["Ngưỡng điểm (chạy end-to-end)", f"{m['score_threshold']} (cosine, `SCORE_THRESHOLD` trong `.env`)"],
        ["Các giai đoạn đã chạy", ", ".join(m["stages"]) or "–"],
    ]
    env_model, env_thinking = m.get("env_llm_model"), m.get("env_llm_thinking")
    if (env_model, env_thinking) != (m["llm_model"], m["llm_thinking"]) and env_model:
        rows.append(["Lưu ý", f"`.env` đặt `{env_model}` (thinking `{env_thinking or 'mặc định'}`); lần chạy này ghi đè bằng "
                              "`--model`/`--thinking` cho riêng tiến trình này, `.env` không bị sửa"])
    return table(["Mục", "Giá trị"], rows)


def _retrieval(results: dict, s: dict) -> list[str]:
    ks = s["ks"]
    out = ["## 3. Truy xuất (retrieval)", ""]
    if not s["overall"]:
        return out + ["Không có dữ liệu truy xuất.", ""]
    n = len(s["scores"])
    out += [
        f"Đo trên {n} câu có đáp án trong tài liệu (single + multi). Một chunk được coi là đúng nếu **trang** của nó "
        "nằm trong `relevant_pages`. Precision@K = số chunk đúng trong K kết quả đầu / K. "
        "Recall@K = số trang đúng tìm thấy / số trang đúng. MRR tính trong top-"
        f"{max(ks)}. Truy xuất chạy **không có ngưỡng điểm** để thấy điểm thật.",
        "",
        "### 3.1 Tổng thể",
        "",
        table(["Chỉ số"] + [f"K={k}" for k in ks], [
            ["Precision@K"] + [f3(s["overall"].precision[k]) for k in ks],
            ["Precision@K tối đa có thể đạt (trần)"] + [f3(s["overall"].ceiling[k]) for k in ks],
            ["Recall@K"] + [f3(s["overall"].recall[k]) for k in ks],
            ["Hit-rate@K"] + [f3(s["overall"].hit[k]) for k in ks],
            [f"MRR (top-{max(ks)})"] + [f3(s["overall"].mrr)] + ["" for _ in ks[1:]],
        ]),
        "",
        "Lưu ý về *trần*: nhãn ở mức trang, mà mỗi trang chỉ có vài chunk; một câu có đáp án ở trang chỉ có 2 chunk "
        "thì Precision@5 không thể vượt 2/5 dù truy xuất hoàn hảo. Vì vậy hãy đọc Precision cùng với cột trần.",
        "",
        "### 3.2 Theo loại câu hỏi",
        "",
        table(["Loại", "n"] + [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + [f"Hit@{k}" for k in ks] + ["MRR"], [
            [TYPE_VI[t], sum(1 for q in results["questions"] if q["type"] == t and q["id"] in s["scores"])]
            + [f3(v.precision[k]) for k in ks] + [f3(v.recall[k]) for k in ks] + [f3(v.hit[k]) for k in ks] + [f3(v.mrr)]
            for t, v in s["by_type"].items()
        ]),
        "",
        "### 3.3 Từng câu có đáp án",
        "",
    ]
    rows = []
    for q in results["questions"]:
        sc = s["scores"].get(q["id"])
        if not sc:
            continue
        k = max(ks)
        retrieved = ", ".join(f"{'?' if h['page'] is None else h['page']} ({h['score']:.3f})" for h in q["hits"])
        rows.append([q["id"], q["type"], pages_text(q["relevant_pages"]), retrieved,
                     f3(sc.precision[ks[0]]), f3(sc.precision[k]), f3(sc.recall[k]), f3(sc.mrr)])
    out += [table(["Câu", "Loại", "Trang đúng", "Top-5: trang (điểm)", f"P@{ks[0]}", f"P@{max(ks)}",
                   f"R@{max(ks)}", "RR"], rows), ""]
    out += ["### 3.4 Câu không có đáp án (insufficient / unrelated): điểm top-1 thay cho Precision/Recall", ""]
    rows = []
    for q in results["questions"]:
        if q["expect_fallback"] and q["hits"]:
            top = max(q["hits"], key=lambda h: h["score"])
            rows.append([q["id"], q["type"], q["question"], f"{top['score']:.4f}", "?" if top["page"] is None else top["page"],
                         top["chunk_index"]])
    out += [table(["Câu", "Loại", "Câu hỏi", "Điểm top-1", "Trang top-1", "Chunk top-1"], rows), ""]
    return out


def _threshold(results: dict, s: dict) -> list[str]:
    out = ["## 4. Nghiên cứu ngưỡng điểm", ""]
    if not s["study"]:
        return out + ["Không đủ dữ liệu (cần cả câu có đáp án và câu không có đáp án).", ""]
    a, f = s["answerable"], s["fallback"]
    sep, rec, current = s["separation"], s["recommendation"], results["meta"]["score_threshold"]
    top1 = s["top1"]

    def stats(group: list[dict]) -> list[Any]:
        values = [top1[q["id"]] for q in group]
        return [len(values), f3(min(values)), f3(median(values)), f3(max(values))]

    def where(group: list[dict], pick: Any) -> str:
        return pick(group, key=lambda q: top1[q["id"]])["id"]

    out += [
        f"Dùng điểm top-1 của {len(a) + len(f)} câu. Một câu *qua cổng* khi có ít nhất một chunk đạt điểm ≥ ngưỡng, "
        "tức là khi top-1 ≥ ngưỡng.",
        "",
        table(["Nhóm", "n", "min", "median", "max"], [["Có đáp án (single + multi)"] + stats(a),
                                                     ["Không có đáp án (insufficient + unrelated)"] + stats(f)]),
        "",
    ]
    for t in ("insufficient", "unrelated"):
        group = [q for q in f if q["type"] == t]
        if group:
            out.append(f"- {TYPE_VI[t]}: " + ", ".join(f"{q['id']} = {top1[q['id']]:.4f}" for q in group))
    out += [""]
    if sep.overlaps:
        out.append(f"**Hai lớp CHỒNG LẤN**: câu không có đáp án cao nhất ({where(f, max)}) có top-1 = {sep.max_fallback:.4f}, "
                   f"không thấp hơn câu có đáp án thấp nhất ({where(a, min)}) = {sep.min_answerable:.4f} "
                   f"(chênh {sep.gap:+.4f}). Không có ngưỡng nào tách hoàn toàn hai lớp.")
    else:
        out.append(f"Hai lớp **tách được** nhưng khe hẹp: câu không có đáp án cao nhất ({where(f, max)}) = {sep.max_fallback:.4f}, "
                   f"câu có đáp án thấp nhất ({where(a, min)}) = {sep.min_answerable:.4f}; khoảng cách = {sep.gap:.4f}. "
                   "Mọi ngưỡng trong khoảng này đều không có lỗi trên 20 câu này.")
    out += ["", "### 4.1 Bảng ngưỡng 0.55 → 0.80", "",
            table(["Ngưỡng", f"Chặn nhầm (/{len(a)})", f"Lọt qua nhầm (/{len(f)})", "Tổng lỗi"],
                  [[f"{r.threshold:.2f}" + (" ← hiện tại" if abs(r.threshold - current) < 1e-9 else ""),
                    r.wrongly_blocked, r.wrongly_passed, r.errors] for r in s["study"]]),
            "",
            "*Chặn nhầm* = câu có đáp án nhưng top-1 < ngưỡng (người dùng nhận fallback dù tài liệu có đáp án). "
            "*Lọt qua nhầm* = câu không có đáp án nhưng top-1 ≥ ngưỡng (chunk được gửi cho LLM; LLM vẫn có thể từ chối).",
            "", "### 4.2 Khuyến nghị", ""]
    row_now = next((r for r in s["study"] if abs(r.threshold - current) < 1e-9), None)
    if row_now:
        out.append(f"- Ngưỡng hiện tại {current}: chặn nhầm {row_now.wrongly_blocked}, lọt qua nhầm {row_now.wrongly_passed}.")
    run = rec.tied_run
    if len(run) == 1:
        out.append(f"- Ít lỗi nhất ({rec.errors} lỗi): chỉ đúng một điểm của lưới, **{rec.threshold:.2f}**.")
    else:
        out.append(f"- Ít lỗi nhất ({rec.errors} lỗi): các ngưỡng {run[0]:.2f} – {run[-1]:.2f}; chọn giữa dải = **{rec.threshold:.2f}**.")
    if rec.max_without_blocking is not None:
        passed = next(r.wrongly_passed for r in s["study"] if r.threshold == rec.max_without_blocking)
        out.append(f"- Ngưỡng cao nhất không chặn nhầm câu nào: {rec.max_without_blocking:.2f} "
                   f"(khi đó {passed}/{len(f)} câu không có đáp án vẫn lọt qua).")
    else:
        out.append("- Mọi ngưỡng trong lưới đều chặn nhầm ít nhất một câu có đáp án.")
    nearest_id, nearest, margin = s["margin"]
    out.append(f"- Biên an toàn của {rec.threshold:.2f}: điểm top-1 gần nhất là {nearest_id} = {nearest:.4f} (cách {margin:.4f})."
               + (" Biên nhỏ hơn 0.01 nghĩa là chỉ cần câu hỏi diễn đạt khác đi một chút là kết quả đổi: đây là một điểm"
                  " nằm sát mép, không phải một vùng an toàn." if margin < 0.01 else ""))
    out += ["- **Cảnh báo**: khuyến nghị được chọn trên chính 20 câu này (và chỉ một tài liệu) nên lạc quan. "
            "Nó cho biết dải hợp lý, không phải con số đã được kiểm chứng trên dữ liệu mới.", ""]

    if s["two"]:
        out += ["### 4.3 Thử ý tưởng hai ngưỡng (chỉ đo, không đổi `rag.py`)", "",
                "Cổng (*gate*) = top-1 phải đạt mức này thì mới trả lời; *include* = chỉ chunk có điểm ≥ include mới vào prompt "
                "(include ≤ gate; include = gate là cách làm hiện tại). *Precision/Recall chunk gửi đi* tính ở mức trang "
                "trên các câu đã qua cổng (precision) và trên mọi câu có đáp án, câu bị chặn tính 0 (recall).", "",
                table(["Gate", "Include", "Chặn nhầm", "Lọt qua nhầm", "Chunk gửi/câu", "Precision chunk gửi", "Recall chunk gửi",
                       "Chunk nhiễu/câu lọt"],
                      [[f"{r.gate:.2f}", f"{r.include:.2f}", r.wrongly_blocked, r.wrongly_passed, f3(r.chunks_sent),
                        f3(r.include_precision), f3(r.include_recall), f3(r.noise_chunks)] for r in s["two"]]),
                "", "Giới hạn: top-5 là trần của số chunk gửi; recall chỉ tính trong 5 chunk đã truy xuất.", ""]
    return out


def _end_to_end(results: dict, s: dict) -> list[str]:
    qs = results["questions"]
    out = ["## 5. End-to-end (RagPipeline.answer)", ""]
    if "end_to_end" not in results["meta"]["stages"]:
        return out + ["Giai đoạn này không được chạy (`--skip-llm`).", ""]
    ans = [q for q in qs if not q["expect_fallback"]]
    fb = [q for q in qs if q["expect_fallback"]]
    count = lambda group, *names: sum(1 for q in group if q["outcome"] in names)  # noqa: E731
    correct, partial, incorrect = (count(ans, n) for n in ("correct", "partially_correct", "incorrect"))
    blocked = count(ans, "wrongly_blocked")
    good_fb = count(fb, "correct_fallback", "correct_refusal")
    out += [
        f"Ngưỡng {results['meta']['score_threshold']}, top_k = {results['meta']['top_k']}, không lọc tài liệu.",
        "",
        "### 5.1 Câu có đáp án",
        "",
        table(["Kết quả", f"Số câu (/{len(ans)})"], [
            ["Đúng", correct], ["Đúng một phần", partial], ["Sai", incorrect],
            ["Bị ngưỡng chặn nhầm (fallback)", blocked], ["Chưa chấm / thiếu", count(ans, "unjudged") + sum(1 for q in ans if not q["outcome"])],
        ]),
        "",
        f"Tỉ lệ đúng nghiêm ngặt (chỉ 'Đúng', câu bị chặn tính là sai): {correct}/{len(ans)}. "
        f"Đúng hoặc đúng một phần: {correct + partial}/{len(ans)}.",
        "",
        "### 5.2 Câu không có đáp án (đúng = fallback, hoặc LLM từ chối đúng)",
        "",
        table(["Kết quả", f"Số câu (/{len(fb)})"], [
            ["Fallback đúng (không gọi LLM)", count(fb, "correct_fallback")],
            ["LLM được gọi và từ chối đúng", count(fb, "correct_refusal")],
            ["Trả lời nhầm như thể tài liệu có", count(fb, "wrongly_answered")],
            ["Chưa chấm / thiếu", count(fb, "unjudged") + sum(1 for q in fb if not q["outcome"])],
        ]),
        "",
        f"Xử lý đúng: {good_fb}/{len(fb)}.",
        "",
    ]
    judged = [q for q in qs if q["judge"]]
    if judged:
        faithful = sum(1 for q in judged if q["judge"]["faithful"])
        nonref = [q for q in judged if not q["judge"]["is_refusal"]]
        out += ["### 5.3 Trung thực (faithfulness) và trích dẫn", "",
                f"- Judge đã chấm {len(judged)} câu có gọi LLM. Trung thực (không có khẳng định nào thiếu căn cứ trong chunk): "
                f"**{faithful}/{len(judged)}**; riêng các câu không phải từ chối: "
                f"{sum(1 for q in nonref if q['judge']['faithful'])}/{len(nonref)}."]
        answered = [q for q in qs if q["e2e"] and not q["e2e"]["used_fallback"]]
        cited_any = sum(1 for q in answered if any(x["cited"] for x in q["e2e"]["sources"]))
        out.append(f"- Câu trả lời có ít nhất một trích dẫn hợp lệ [n]: {cited_any}/{len(answered)}.")
        on_target = [q for q in answered if not q["expect_fallback"]]
        hit = sum(1 for q in on_target if any(x["cited"] and x["page"] in q["relevant_pages"] for x in q["e2e"]["sources"]))
        out.append(f"- Câu có đáp án mà ít nhất một nguồn được trích nằm ở trang đúng: {hit}/{len(on_target)}.")
        bad = sum(len(q["e2e"]["invalid_citations"]) for q in answered)
        out.append(f"- Số trích dẫn tới nguồn không tồn tại (đã bị loại khỏi câu trả lời): {bad}.")
        out.append("")
    out += ["### 5.4 Từng câu", ""]
    rows = []
    for q in qs:
        e, j = q["e2e"], q["judge"]
        rows.append([
            q["id"], q["type"], OUTCOME_VI.get(q["outcome"], "THIẾU"),
            "–" if not e else ("có" if e["used_fallback"] else "không"),
            f3(s["top1"].get(q["id"])),
            "–" if not e else pages_text([x["page"] for x in e["sources"]]),
            "–" if not e else pages_text([x["page"] for x in e["sources"] if x["cited"]]),
            "–" if not j else ("có" if j["faithful"] else "KHÔNG"),
            secs(e["latency"]["total_ms"]) if e else "–", "–" if not e else e["retries"],
        ])
    out += [table(["Câu", "Loại", "Kết quả", "Fallback", "Top-1", "Trang gửi LLM", "Trang được trích", "Trung thực",
                   "Tổng (s)", "Retry"], rows), ""]
    return out


def _latency(s: dict) -> list[str]:
    out = ["## 6. Độ trễ", ""]
    lat = s["latency"]
    if not lat:
        return out + ["Không có dữ liệu.", ""]
    rows = [[f"Tất cả ({lat['n']} câu)", secs(lat["median"]), secs(lat["p95"]), secs(lat["max"])]]
    if lat["llm_n"]:
        rows.append([f"Chỉ câu có gọi LLM ({lat['llm_n']} câu)", secs(lat["llm_median"]), secs(lat["llm_p95"]), secs(lat["llm_max"])])
    out += [table(["Nhóm (total_ms, giây)", "Median", "p95", "Max"], rows), "",
            f"p95 tính bằng nội suy tuyến tính giữa hai hạng gần nhất (như NumPy). Câu chậm nhất: {lat['slowest']}. "
            f"Số câu có retry: {lat['retried']} (tổng {lat['retries']} lần retry; thời gian chờ backoff nằm trong total_ms).",
            ]
    if lat["fallback_median"] is not None:
        out.append(f"Câu fallback (không gọi LLM) có median {secs(lat['fallback_median'])} s, chủ yếu là một lần embed câu hỏi.")
    out += ["", "Đuôi chậm của LLM free tier là vấn đề đã biết (KI-18 trong `docs/known_issues.md`): với N nhỏ, "
            "p95 gần như chỉ là câu chậm nhất, nên đừng đọc nó như một con số ổn định.", ""]
    return out


def _judge_check(results: dict, review: dict | None) -> list[str]:
    out = ["## 7. Kiểm tra độ tin cậy của judge", "",
           "Judge là một LLM nên **không phải sự thật tuyệt đối**: có thể quá dễ dãi, thiên vị câu dài, hoặc thiên vị văn phong "
           "của chính mô hình (ở đây judge và mô hình trả lời "
           + ("cùng một mô hình" if results["meta"]["judge_model"] == results["meta"]["llm_model"] else "khác mô hình") + "). "
           "Hai cách kiểm tra bên dưới đều chưa thay được việc người thật đọc lại.", ""]
    by_id = {q["id"]: q for q in results["questions"]}
    sample = [i for i in results["meta"].get("sample_ids", []) if i in by_id and by_id[i]["judge"] and by_id[i]["e2e"]]
    if sample:
        out += [f"### 7.1 {len(sample)} verdict chọn ngẫu nhiên để bạn tự kiểm tra (seed {results['meta']['seed']})", ""]
        for qid in sample:
            q, j = by_id[qid], by_id[qid]["judge"]
            out += [f"**{qid}** — {q['question']}", "",
                    f"- Đáp án chuẩn: {q['reference_answer']}",
                    f"- Câu trả lời của hệ thống: {q['e2e']['answer'].replace(chr(10), ' ')}",
                    f"- Judge: **{j['correctness']}**, từ chối = {j['is_refusal']}, trung thực = {j['faithful']}. {j['notes']}"]
            out += [f"  - Khẳng định thiếu căn cứ: {c}" for c in j["unsupported_claims"]]
            out.append("")
    number = "7.2" if sample else "7.1"
    if not review:
        return out + [f"### {number} Ý kiến thứ hai", "", "Chưa có (`evaluation/claude_review.json` không tồn tại).", ""]
    verdicts = review["verdicts"]
    pairs = [(by_id[i], v) for i, v in verdicts.items() if i in by_id and by_id[i]["judge"]]
    same_c = [(q, v) for q, v in pairs if q["judge"]["correctness"] == v["correctness"]]
    same_f = [(q, v) for q, v in pairs if q["judge"]["faithful"] == v["faithful"]]
    out += [f"### {number} Ý kiến thứ hai: {review['reviewer']}", "",
            review.get("note", ""), "",
            f"Trên {len(pairs)} câu judge đã chấm: đồng ý về mức đúng/sai {len(same_c)}/{len(pairs)}, đồng ý về trung thực "
            f"{len(same_f)}/{len(pairs)}.", ""]
    diffs = [(q, v) for q, v in pairs if q["judge"]["correctness"] != v["correctness"] or q["judge"]["faithful"] != v["faithful"]]
    if diffs:
        out += ["Các câu hai bên **không đồng ý**:", "",
                table(["Câu", "Judge: đúng/sai", "Judge: trung thực", "Ý kiến thứ hai: đúng/sai", "Ý kiến thứ hai: trung thực", "Lý do (ý kiến thứ hai)"],
                      [[q["id"], q["judge"]["correctness"], q["judge"]["faithful"], v["correctness"], v["faithful"], v["comment"]]
                       for q, v in diffs]), ""]
    else:
        out += ["Không có câu nào bất đồng về nhãn. Điều đó chưa chứng minh judge đúng: hai LLM có thể cùng mắc một kiểu sai.", ""]
    close = [(q, v) for q, v in pairs if v.get("borderline")]
    if close:
        out += ["Các câu mà ý kiến thứ hai coi là **ranh giới** (nhãn có thể đổi tùy cách đọc tiêu chí):", "",
                *[f"- **{q['id']}** (judge: {q['judge']['correctness']}; ý kiến thứ hai: {v['correctness']}): {v['comment']}"
                  for q, v in close], ""]
    return out


def _limitations(results: dict) -> list[str]:
    m = results["meta"]
    n = len(results["questions"])
    n_ans = sum(1 for q in results["questions"] if not q["expect_fallback"])
    return [
        "## 10. Hạn chế", "",
        f"- **Một tài liệu duy nhất, {m['num_pages']} trang, {m['num_chunks']} chunk**: kết quả không nói gì về tài liệu dài, "
        "nhiều tài liệu cùng lúc, hay tài liệu có bảng/hình.",
        f"- **N nhỏ ({n} câu, {n_ans} có đáp án)**: mỗi câu đổi chiều làm tỉ lệ đổi {100 / n:.0f}–{100 / n_ans:.0f} điểm phần trăm; "
        "không có khoảng tin cậy nào đáng tin ở cỡ này. Đây là kiểm tra hồi quy và phát hiện lỗi, không phải benchmark.",
        "- **Nhãn ở mức trang**: một chunk ở đúng trang nhưng không chứa đáp án vẫn được tính là đúng; ngược lại đáp án nằm ở "
        "nhiều trang thì nhãn có thể thiếu. Precision@K bị chặn trên bởi số chunk của trang (xem cột trần).",
        "- **Judge thiên lệch**: LLM chấm LLM; xem mục 7. Nhãn tham chiếu do chủ dự án tự viết từ PDF, cũng có thể sai sót.",
        "- **Ngưỡng được chọn và đánh giá trên cùng một bộ câu hỏi**; ngoài ra một số câu (PgBouncer, `pg_upgrade`, sharding, nấu phở) "
        "trùng với các câu đã dùng khi đặt ngưỡng tạm 0.65 (KI-17), nên ngưỡng đó không hoàn toàn độc lập với bộ test.",
        "- **LLM không tất định** (temperature 0.2) và free tier có đuôi chậm: chạy lại không cache có thể cho kết quả khác một chút. "
        "Cache trong `evaluation/cache/` giữ nguyên lần chạy này.",
        "- Không đo: ảnh hưởng của header/footer lặp lại (KI-01), chunk bìa/mục lục (KI-14) một cách riêng biệt — chỉ quan sát được "
        "qua danh sách chunk truy xuất ở mục 3.3.",
        "",
    ]


def _reproduce() -> list[str]:
    return [
        "## 11. Cách chạy lại", "",
        "```powershell",
        "docker compose up -d                                   # Qdrant",
        ".venv\\Scripts\\python.exe evaluation/evaluate.py        # chạy đủ 4 giai đoạn (dùng cache nếu có)",
        ".venv\\Scripts\\python.exe evaluation/evaluate.py --skip-llm      # chỉ truy xuất + nghiên cứu ngưỡng",
        ".venv\\Scripts\\python.exe evaluation/evaluate.py --report-only   # dựng lại report.md từ results.json",
        ".venv\\Scripts\\python.exe -m pytest tests/test_eval_metrics.py tests/test_eval_judge.py -q",
        "```",
        "",
        "Mọi lời gọi API được cache trong `evaluation/cache/` theo hash của đầu vào, nên chạy lại không tốn quota và cho đúng "
        "các số này; xóa thư mục đó để chạy lại từ đầu. Kết quả thô nằm ở `evaluation/results.json`.",
        "",
    ]


# ---------------------------------------------------------------------- the report


def render_report(results: dict, analysis: str | None, review: dict | None) -> str:
    s = summarize(results)
    missing = find_missing(results)
    meta = results["meta"]
    out = [
        "# Báo cáo đánh giá — AI Learning Assistant (RAG)", "",
        f"*Sinh tự động bởi `evaluation/evaluate.py` từ `evaluation/results.json` (lần chạy ngày {meta['date']}). "
        "Mọi con số trong bảng đến từ lần chạy này; mục 9 do người viết tay sau khi đọc kết quả.*", "",
    ]
    if any(missing.values()):
        out += ["> **Dữ liệu chưa đầy đủ.** " + "; ".join(f"giai đoạn `{stage}` thiếu: {', '.join(ids)}"
                                                         for stage, ids in missing.items() if ids)
                + ". Các câu này không được điền thay thế; chạy lại để tiếp tục (phần đã làm được lấy từ cache).", ""]
    out += ["## 1. Tóm tắt", ""]
    if s["overall"]:
        k = max(s["ks"])
        out.append(f"- Truy xuất ({len(s['scores'])} câu có đáp án): Hit@{k} = {f3(s['overall'].hit[k])}, "
                   f"Recall@{k} = {f3(s['overall'].recall[k])}, MRR = {f3(s['overall'].mrr)}, "
                   f"Precision@{s['ks'][0]} = {f3(s['overall'].precision[s['ks'][0]])} (trần {f3(s['overall'].ceiling[s['ks'][0]])}).")
    if s["separation"]:
        sep = s["separation"]
        out.append(f"- Ngưỡng: hai lớp điểm top-1 {'**chồng lấn**' if sep.overlaps else 'tách được (khe hẹp)'} "
                   f"(khoảng cách {sep.gap:+.4f}); ngưỡng ít lỗi nhất trên lưới là {s['recommendation'].threshold:.2f} "
                   f"nhưng điểm top-1 gần nhất ({s['margin'][0]}) chỉ cách {s['margin'][2]:.4f}; hiện tại {meta['score_threshold']} (mục 4.2).")
    if "end_to_end" in meta["stages"]:
        ans = [q for q in results["questions"] if not q["expect_fallback"]]
        fb = [q for q in results["questions"] if q["expect_fallback"]]
        c = Counter(q["outcome"] for q in ans)
        good_fb = sum(1 for q in fb if q["outcome"] in ("correct_fallback", "correct_refusal"))
        out.append(f"- End-to-end: câu có đáp án đúng {c['correct']}/{len(ans)}, đúng một phần {c['partially_correct']}, "
                   f"sai {c['incorrect']}, bị chặn nhầm {c['wrongly_blocked']}; câu không có đáp án xử lý đúng {good_fb}/{len(fb)}.")
    if s["latency"]:
        lat = s["latency"]
        out.append(f"- Độ trễ tổng: median {secs(lat['median'])} s, p95 {secs(lat['p95'])} s, max {secs(lat['max'])} s.")
    out += ["", "## 2. Thiết lập", "", _setup(results), ""]
    out += _retrieval(results, s) + _threshold(results, s) + _end_to_end(results, s) + _latency(s)
    out += _judge_check(results, review)
    flags = flag_questions(results, s)
    out += ["## 8. Câu sai hoặc đáng ngờ (phát hiện tự động từ số đo)", ""]
    out += ([f"- **{qid}**: " + "; ".join(why) for qid, why in flags] if flags else ["Không có câu nào bị đánh dấu."])
    out += ["", "## 9. Phân tích nguyên nhân và đề xuất cải thiện", ""]
    out += [analysis.strip() if analysis else "*Chưa có (`evaluation/analysis_vi.md` chưa được viết).*", ""]
    out += _limitations(results) + _reproduce()
    return "\n".join(out).rstrip() + "\n"
