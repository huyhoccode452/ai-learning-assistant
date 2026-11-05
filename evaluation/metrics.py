"""Pure evaluation functions: retrieval metrics, threshold study, outcome classification, statistics.

No network and no file access in this module, so every function is tested with tiny fixtures whose
expected values were computed by hand (tests/test_eval_metrics.py).

Conventions (they matter when you read the numbers):
  * Relevance is PAGE level: a retrieved chunk is relevant when its page is in `relevant_pages`.
    A chunk with page=None (DOCX/TXT) is never relevant.
  * Precision@K = relevant chunks in the first K results / K. The divisor is K even when fewer than
    K results exist, so a short result list cannot look better than a full one.
  * Recall@K = distinct relevant pages found in the first K results / number of relevant pages.
  * A chunk passes the score threshold when score >= threshold, the same rule as `RagPipeline`.
"""

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

Pages = Sequence[int | None]


# ---------------------------------------------------------------------- retrieval metrics


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def precision_at_k(retrieved: Pages, relevant: Collection[int], k: int) -> float:
    """Relevant chunks among the first k results, divided by k."""
    _check_k(k)
    return sum(1 for page in retrieved[:k] if page is not None and page in relevant) / k


def recall_at_k(retrieved: Pages, relevant: Collection[int], k: int) -> float:
    """Distinct relevant pages that appear in the first k results / number of relevant pages."""
    _check_k(k)
    wanted = set(relevant)
    if not wanted:
        raise ValueError("recall is undefined when there are no relevant pages")
    found = {page for page in retrieved[:k] if page in wanted}
    return len(found) / len(wanted)


def hit_at_k(retrieved: Pages, relevant: Collection[int], k: int) -> float:
    """1.0 if at least one of the first k results is relevant, else 0.0."""
    _check_k(k)
    return 1.0 if any(page is not None and page in relevant for page in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Pages, relevant: Collection[int], k: int) -> float:
    """1 / rank of the first relevant result within the first k, 0.0 if there is none."""
    _check_k(k)
    for rank, page in enumerate(retrieved[:k], start=1):
        if page is not None and page in relevant:
            return 1 / rank
    return 0.0


def precision_ceiling(corpus_pages: Pages, relevant: Collection[int], k: int) -> float:
    """The best Precision@K any retriever could reach: a question whose relevant pages hold only
    2 chunks cannot get more than 2/K, however good the search is.

    `corpus_pages` = the page of every chunk in the index.
    """
    _check_k(k)
    available = sum(1 for page in corpus_pages if page is not None and page in relevant)
    return min(k, available) / k


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """All retrieval metrics of one question (or their average over many questions)."""

    precision: dict[int, float]  # by K
    recall: dict[int, float]
    hit: dict[int, float]
    ceiling: dict[int, float]
    mrr: float  # reciprocal rank within the largest K


def score_retrieval(
    retrieved: Pages, relevant: Collection[int], ks: Sequence[int], corpus_pages: Pages
) -> RetrievalScores:
    if not ks:
        raise ValueError("ks must not be empty")
    return RetrievalScores(
        precision={k: precision_at_k(retrieved, relevant, k) for k in ks},
        recall={k: recall_at_k(retrieved, relevant, k) for k in ks},
        hit={k: hit_at_k(retrieved, relevant, k) for k in ks},
        ceiling={k: precision_ceiling(corpus_pages, relevant, k) for k in ks},
        mrr=reciprocal_rank(retrieved, relevant, max(ks)),
    )


def average_scores(items: Sequence[RetrievalScores]) -> RetrievalScores:
    """Macro average: every question counts the same, however many chunks it has."""
    if not items:
        raise ValueError("cannot average zero questions")
    ks = list(items[0].precision)

    def avg(field: str) -> dict[int, float]:
        return {k: mean([getattr(item, field)[k] for item in items]) for k in ks}

    return RetrievalScores(
        precision=avg("precision"),
        recall=avg("recall"),
        hit=avg("hit"),
        ceiling=avg("ceiling"),
        mrr=mean([item.mrr for item in items]),
    )


# ---------------------------------------------------------------------- statistics


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        raise ValueError("mean of an empty sequence")
    return sum(numbers) / len(numbers)


def percentile(values: Iterable[float], p: float) -> float:
    """p-th percentile (0..100) with linear interpolation between the closest ranks (the NumPy
    default). Example: for [1, 2, 3, 4], p=50 -> 2.5 and p=95 -> 3.85."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be between 0 and 100, got {p}")
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def median(values: Iterable[float]) -> float:
    return percentile(values, 50)


# ---------------------------------------------------------------------- threshold study


def frange(start: float, stop: float, step: float) -> list[float]:
    """start, start+step, ... up to and including stop (rounded, so 0.55 + 0.01*3 is 0.58, not
    0.5800000000000001)."""
    count = round((stop - start) / step)
    return [round(start + i * step, 6) for i in range(count + 1)]


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    threshold: float
    wrongly_blocked: int  # answerable questions whose best score is below the threshold
    wrongly_passed: int  # insufficient/unrelated questions whose best score reaches it

    @property
    def errors(self) -> int:
        return self.wrongly_blocked + self.wrongly_passed


def threshold_study(
    answerable_top1: Sequence[float], fallback_top1: Sequence[float], thresholds: Iterable[float]
) -> list[ThresholdRow]:
    """For each threshold: how many answerable questions would be blocked and how many questions
    that should fall back would pass. Only the best (top-1) score of a question matters, because
    a question passes as soon as ONE chunk reaches the threshold."""
    return [
        ThresholdRow(
            threshold=t,
            wrongly_blocked=sum(1 for s in answerable_top1 if s < t),
            wrongly_passed=sum(1 for s in fallback_top1 if s >= t),
        )
        for t in thresholds
    ]


@dataclass(frozen=True, slots=True)
class Separation:
    max_fallback: float  # highest top-1 among questions that should fall back
    min_answerable: float  # lowest top-1 among answerable questions
    gap: float  # min_answerable - max_fallback; <= 0 means the classes overlap
    overlaps: bool


def class_separation(answerable_top1: Sequence[float], fallback_top1: Sequence[float]) -> Separation:
    if not answerable_top1 or not fallback_top1:
        raise ValueError("need at least one score in each class")
    max_fallback, min_answerable = max(fallback_top1), min(answerable_top1)
    gap = min_answerable - max_fallback
    return Separation(max_fallback, min_answerable, gap, overlaps=gap <= 0)


@dataclass(frozen=True, slots=True)
class Recommendation:
    threshold: float  # middle of the longest run of thresholds with the fewest total errors
    errors: int
    tied_run: tuple[float, ...]  # that run
    max_without_blocking: float | None  # highest grid threshold that blocks no answerable question


def recommend_threshold(rows: Sequence[ThresholdRow]) -> Recommendation:
    """Fewest (blocked + passed) errors; ties -> the middle of the longest run of neighbouring
    grid points, so the choice sits away from the edge of the good range."""
    if not rows:
        raise ValueError("no threshold rows")
    best = min(row.errors for row in rows)
    runs: list[list[int]] = []  # indexes into rows
    for index, row in enumerate(rows):
        if row.errors != best:
            continue
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    longest = max(runs, key=len)  # max keeps the first of equally long runs
    chosen = longest[(len(longest) - 1) // 2]
    unblocked = [row.threshold for row in rows if row.wrongly_blocked == 0]
    return Recommendation(
        threshold=rows[chosen].threshold,
        errors=best,
        tied_run=tuple(rows[i].threshold for i in longest),
        max_without_blocking=max(unblocked) if unblocked else None,
    )


# ---------------------------------------------------------------------- two-threshold measurement


@dataclass(frozen=True, slots=True)
class ScoredHit:
    score: float
    page: int | None


@dataclass(frozen=True, slots=True)
class TwoThresholdRow:
    """gate = the question may be answered only if its best score reaches this.
    include = a chunk goes into the prompt only if its own score reaches this (include <= gate).
    include == gate is today's single threshold."""

    gate: float
    include: float
    wrongly_blocked: int
    wrongly_passed: int
    chunks_sent: float | None  # mean chunks sent, answerable questions that passed the gate
    include_precision: float | None  # same questions: relevant chunks sent / chunks sent
    include_recall: float  # ALL answerable questions: relevant pages sent / relevant pages (blocked = 0)
    noise_chunks: float | None  # mean chunks sent for should-fall-back questions that passed the gate


def two_threshold_row(
    answerable: Sequence[tuple[Sequence[ScoredHit], Collection[int]]],
    fallback: Sequence[Sequence[ScoredHit]],
    gate: float,
    include: float,
) -> TwoThresholdRow:
    """Measure one (gate, include) pair. `answerable` = (hits, relevant pages) per question."""
    if include > gate:
        raise ValueError(f"include ({include}) must not be above gate ({gate})")

    def top1(hits: Sequence[ScoredHit]) -> float:
        return max((h.score for h in hits), default=-math.inf)

    def sent(hits: Sequence[ScoredHit]) -> list[ScoredHit]:
        return [h for h in hits if h.score >= include]

    blocked = 0
    sizes: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for hits, relevant in answerable:
        if top1(hits) < gate:
            blocked += 1
            recalls.append(0.0)
            continue
        chosen = sent(hits)
        sizes.append(len(chosen))
        precisions.append(sum(1 for h in chosen if h.page in relevant) / len(chosen))
        recalls.append(len({h.page for h in chosen if h.page in relevant}) / len(set(relevant)))
    passed_noise = [len(sent(hits)) for hits in fallback if top1(hits) >= gate]
    return TwoThresholdRow(
        gate=gate,
        include=include,
        wrongly_blocked=blocked,
        wrongly_passed=len(passed_noise),
        chunks_sent=mean(sizes) if sizes else None,
        include_precision=mean(precisions) if precisions else None,
        include_recall=mean(recalls) if recalls else 0.0,
        noise_chunks=mean(passed_noise) if passed_noise else None,
    )


# ---------------------------------------------------------------------- end-to-end outcome


class Outcome(StrEnum):
    # answerable question
    CORRECT = "correct"
    PARTIAL = "partially_correct"
    INCORRECT = "incorrect"
    WRONGLY_BLOCKED = "wrongly_blocked"  # fell back although the document has the answer
    # question that should not be answered from the document
    CORRECT_FALLBACK = "correct_fallback"  # pipeline fell back, LLM not called
    CORRECT_REFUSAL = "correct_refusal"  # LLM was called and said the document does not cover it
    WRONGLY_ANSWERED = "wrongly_answered"  # LLM answered as if the document covered it
    UNJUDGED = "unjudged"  # answered by the LLM but no judge verdict (quota ran out)


CORRECTNESS_LABELS = ("correct", "partially_correct", "incorrect")


def classify_outcome(
    expect_fallback: bool, used_fallback: bool, correctness: str | None, is_refusal: bool | None
) -> Outcome:
    """Combine what the pipeline did with the judge's verdict into one outcome."""
    if used_fallback:
        return Outcome.CORRECT_FALLBACK if expect_fallback else Outcome.WRONGLY_BLOCKED
    if correctness is None:
        return Outcome.UNJUDGED
    if correctness not in CORRECTNESS_LABELS:
        raise ValueError(f"unknown correctness label: {correctness!r}")
    if expect_fallback:
        # Only a complete, correct refusal counts; a partial answer is still an answer.
        return (
            Outcome.CORRECT_REFUSAL
            if correctness == "correct" and is_refusal
            else Outcome.WRONGLY_ANSWERED
        )
    return Outcome(correctness)
