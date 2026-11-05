"""Tests for evaluation/metrics.py. Every expected value below was computed by hand (the working is
in the comments), so a wrong formula cannot hide behind a test that repeats the same formula."""

import pytest

from evaluation.metrics import (
    Outcome,
    ScoredHit,
    ThresholdRow,
    average_scores,
    class_separation,
    classify_outcome,
    frange,
    hit_at_k,
    mean,
    median,
    percentile,
    precision_at_k,
    precision_ceiling,
    recall_at_k,
    recommend_threshold,
    reciprocal_rank,
    score_retrieval,
    threshold_study,
    two_threshold_row,
)

RETRIEVED = [5, 2, 5, 9, 4]  # pages of the top-5 chunks, best first


# ---------------------------------------------------------------------- precision / recall / hit / MRR


def test_precision_counts_relevant_chunks_not_pages() -> None:
    # relevant {5}: ranks 1 and 3 are page 5 -> 2 of the first 3, 2 of 5
    assert precision_at_k(RETRIEVED, {5}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RETRIEVED, {5}, 5) == pytest.approx(2 / 5)
    # relevant {2}: only rank 2
    assert precision_at_k(RETRIEVED, {2}, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RETRIEVED, {2}, 5) == pytest.approx(1 / 5)
    # nothing relevant
    assert precision_at_k(RETRIEVED, {7}, 5) == 0.0


def test_precision_divides_by_k_even_when_fewer_results_exist() -> None:
    assert precision_at_k([5], {5}, 3) == pytest.approx(1 / 3)  # not 1/1
    assert precision_at_k([], {5}, 3) == 0.0


def test_precision_ignores_chunks_without_a_page() -> None:
    assert precision_at_k([None, 5], {5}, 2) == pytest.approx(1 / 2)
    assert precision_at_k([None, None], {5}, 2) == 0.0


def test_recall_counts_distinct_relevant_pages() -> None:
    # relevant {4, 6, 8}: only page 4 is retrieved, at rank 5
    assert recall_at_k(RETRIEVED, {4, 6, 8}, 5) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, {4, 6, 8}, 3) == 0.0
    # relevant {9, 5}, results [9, 1, 5]: 1 of 2 pages in the first 2, both in the first 3
    assert recall_at_k([9, 1, 5], {9, 5}, 2) == pytest.approx(1 / 2)
    assert recall_at_k([9, 1, 5], {9, 5}, 3) == 1.0


def test_recall_does_not_count_the_same_page_twice() -> None:
    # three chunks of page 5, relevant {5, 6}: page 5 found once -> 1 of 2, precision is 3/3
    assert recall_at_k([5, 5, 5], {5, 6}, 3) == pytest.approx(1 / 2)
    assert precision_at_k([5, 5, 5], {5, 6}, 3) == 1.0


def test_recall_without_relevant_pages_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, set(), 3)


def test_hit_is_one_when_any_of_the_first_k_is_relevant() -> None:
    assert hit_at_k(RETRIEVED, {4, 6, 8}, 3) == 0.0  # page 4 is only at rank 5
    assert hit_at_k(RETRIEVED, {4, 6, 8}, 5) == 1.0
    assert hit_at_k(RETRIEVED, {5}, 1) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_result() -> None:
    assert reciprocal_rank(RETRIEVED, {5}, 5) == 1.0  # rank 1 (rank 3 does not matter)
    assert reciprocal_rank(RETRIEVED, {2}, 5) == pytest.approx(1 / 2)
    assert reciprocal_rank(RETRIEVED, {9}, 5) == pytest.approx(1 / 4)
    assert reciprocal_rank(RETRIEVED, {4, 6, 8}, 5) == pytest.approx(1 / 5)
    assert reciprocal_rank(RETRIEVED, {4}, 3) == 0.0  # outside the window
    assert reciprocal_rank(RETRIEVED, {7}, 5) == 0.0


@pytest.mark.parametrize("function", [precision_at_k, recall_at_k, hit_at_k, reciprocal_rank])
def test_k_must_be_positive(function) -> None:
    with pytest.raises(ValueError):
        function(RETRIEVED, {5}, 0)


def test_precision_ceiling_is_limited_by_chunks_on_the_relevant_pages() -> None:
    corpus = [1, 1, 2, 2, 2, 3]  # page 2 has 3 chunks, page 1 has 2, page 3 has 1
    assert precision_ceiling(corpus, {2}, 3) == 1.0  # 3 relevant chunks fill K=3
    assert precision_ceiling(corpus, {2}, 5) == pytest.approx(3 / 5)
    assert precision_ceiling(corpus, {1, 3}, 5) == pytest.approx(3 / 5)  # 2 + 1 chunks
    assert precision_ceiling(corpus, {3}, 3) == pytest.approx(1 / 3)
    assert precision_ceiling([None, None, 2], {2}, 3) == pytest.approx(1 / 3)
    assert precision_ceiling([2, 2, 2, 2], {2}, 3) == 1.0  # 4 relevant chunks but only K=3 slots: never above 1


def test_score_retrieval_and_macro_average() -> None:
    corpus = [5, 5, 5, 2, 9, 4]
    a = score_retrieval([5, 2, 5, 9, 4], {5}, (3, 5), corpus)
    b = score_retrieval([1, 2, 3, 4, 6], {2}, (3, 5), corpus)
    # question A: P@3 2/3, P@5 2/5, R 1, hit 1, RR 1, ceiling 1 and 3/5
    assert a.precision == {3: pytest.approx(2 / 3), 5: pytest.approx(2 / 5)}
    assert a.mrr == 1.0 and a.ceiling == {3: 1.0, 5: pytest.approx(3 / 5)}
    # question B: P@3 1/3, P@5 1/5, R 1, hit 1, RR 1/2, ceiling 1/3 and 1/5
    assert b.precision == {3: pytest.approx(1 / 3), 5: pytest.approx(1 / 5)}
    assert b.mrr == pytest.approx(1 / 2)
    mean_scores = average_scores([a, b])
    assert mean_scores.precision == {3: pytest.approx(0.5), 5: pytest.approx(0.3)}  # (2/3+1/3)/2, (2/5+1/5)/2
    assert mean_scores.recall == {3: 1.0, 5: 1.0}
    assert mean_scores.hit == {3: 1.0, 5: 1.0}
    assert mean_scores.mrr == pytest.approx(0.75)  # (1 + 1/2) / 2
    assert mean_scores.ceiling == {3: pytest.approx(2 / 3), 5: pytest.approx(0.4)}  # (1+1/3)/2, (3/5+1/5)/2


def test_mrr_looks_at_the_largest_k_not_the_smallest() -> None:
    # relevant page 4 is only at rank 5: invisible at K=3, but MRR is computed within the largest K (5)
    scores = score_retrieval([5, 2, 5, 9, 4], {4}, (3, 5), [4, 5])
    assert scores.mrr == pytest.approx(1 / 5) and scores.hit == {3: 0.0, 5: 1.0}


def test_average_and_score_need_input() -> None:
    with pytest.raises(ValueError):
        average_scores([])
    with pytest.raises(ValueError):
        score_retrieval(RETRIEVED, {5}, (), [5])


# ---------------------------------------------------------------------- statistics


def test_percentile_interpolates_between_ranks() -> None:
    values = [4, 1, 3, 2]  # unsorted on purpose
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == pytest.approx(2.5)  # position 1.5 -> halfway between 2 and 3
    assert percentile(values, 95) == pytest.approx(3.85)  # position 2.85 -> 3 + 0.85 * (4 - 3)
    assert percentile(values, 100) == 4
    assert percentile([7], 95) == 7


def test_median_mean_and_empty_inputs() -> None:
    assert median([3, 1, 2]) == 2
    assert median([1, 2, 3, 4]) == pytest.approx(2.5)
    assert mean([1, 2, 6]) == 3
    for call in (lambda: mean([]), lambda: percentile([], 50), lambda: percentile([1], 101), lambda: percentile([1], -1)):
        with pytest.raises(ValueError):
            call()


def test_frange_includes_both_ends_without_float_drift() -> None:
    grid = frange(0.55, 0.80, 0.01)
    assert len(grid) == 26 and grid[0] == 0.55 and grid[-1] == 0.8
    assert 0.58 in grid  # exactly 0.58, not 0.5800000000000001
    assert frange(0, 1, 0.5) == [0, 0.5, 1.0]


# ---------------------------------------------------------------------- threshold study


def test_threshold_study_counts_blocked_and_passed() -> None:
    answerable = [0.70, 0.80, 0.90]
    fallback = [0.50, 0.72]
    rows = {r.threshold: r for r in threshold_study(answerable, fallback, [0.6, 0.7, 0.71, 0.72, 0.75, 0.95])}
    # t=0.6 : nothing blocked; 0.72 passes (0.50 does not)                      -> (0, 1)
    assert (rows[0.6].wrongly_blocked, rows[0.6].wrongly_passed) == (0, 1)
    # t=0.7 : a score EQUAL to the threshold passes (>=), like RagPipeline     -> (0, 1)
    assert (rows[0.7].wrongly_blocked, rows[0.7].wrongly_passed) == (0, 1)
    # t=0.71: 0.70 is now blocked; 0.72 still passes                            -> (1, 1)
    assert (rows[0.71].wrongly_blocked, rows[0.71].wrongly_passed) == (1, 1)
    # t=0.72: a fallback question scoring EXACTLY 0.72 passes (>=); 0.70 is blocked -> (1, 1)
    assert (rows[0.72].wrongly_blocked, rows[0.72].wrongly_passed) == (1, 1)
    # t=0.75: 0.70 blocked; 0.72 no longer passes                               -> (1, 0)
    assert (rows[0.75].wrongly_blocked, rows[0.75].wrongly_passed) == (1, 0)
    # t=0.95: everything answerable is blocked                                  -> (3, 0)
    assert (rows[0.95].wrongly_blocked, rows[0.95].wrongly_passed) == (3, 0)
    assert rows[0.71].errors == 2


def test_class_separation_detects_overlap_and_touching_scores() -> None:
    overlapping = class_separation([0.70, 0.80, 0.90], [0.50, 0.72])
    assert overlapping.overlaps and overlapping.max_fallback == 0.72 and overlapping.min_answerable == 0.70
    assert overlapping.gap == pytest.approx(-0.02)
    clean = class_separation([0.70, 0.80], [0.50, 0.60])
    assert not clean.overlaps and clean.gap == pytest.approx(0.10)
    assert class_separation([0.7], [0.7]).overlaps  # equal scores cannot be told apart
    with pytest.raises(ValueError):
        class_separation([], [0.5])


def _rows(spec: list[tuple[float, int, int]]) -> list[ThresholdRow]:
    return [ThresholdRow(t, blocked, passed) for t, blocked, passed in spec]


def test_recommend_picks_the_middle_of_the_best_run() -> None:
    rows = _rows([
        (0.60, 0, 3),  # 3 errors
        (0.61, 0, 1),  # 1
        (0.62, 0, 1),  # 1
        (0.63, 1, 0),  # 1
        (0.64, 1, 0),  # 1
        (0.65, 2, 0),  # 2
        (0.66, 4, 0),  # 4
    ])
    recommendation = recommend_threshold(rows)
    assert recommendation.errors == 1
    assert recommendation.tied_run == (0.61, 0.62, 0.63, 0.64)
    assert recommendation.threshold == 0.62  # run of 4 -> lower middle (index (4-1)//2 = 1)
    assert recommendation.max_without_blocking == 0.62  # 0.63 is the first grid point that blocks one


def test_recommend_prefers_the_longest_run_when_the_minimum_repeats() -> None:
    rows = _rows([(0.60, 0, 1), (0.61, 1, 1), (0.62, 0, 1), (0.63, 0, 1), (0.64, 3, 1)])
    # errors 1, 2, 1, 1, 4 -> runs [0.60] and [0.62, 0.63]
    recommendation = recommend_threshold(rows)
    assert recommendation.tied_run == (0.62, 0.63)
    assert recommendation.threshold == 0.62


def test_recommend_reports_when_every_grid_point_blocks_something() -> None:
    recommendation = recommend_threshold(_rows([(0.6, 1, 0), (0.7, 2, 0)]))
    assert recommendation.max_without_blocking is None
    assert recommendation.threshold == 0.6
    with pytest.raises(ValueError, match="no threshold rows"):
        recommend_threshold([])


# ---------------------------------------------------------------------- two thresholds


def _hits(*pairs: tuple[float, int]) -> list[ScoredHit]:
    return [ScoredHit(score, page) for score, page in pairs]


Q1 = (_hits((0.80, 5), (0.70, 2), (0.60, 5), (0.50, 9)), {5})
Q2 = (_hits((0.66, 3), (0.64, 4)), {3, 4})
F1 = _hits((0.70, 1), (0.62, 2), (0.40, 3))
F2 = _hits((0.50, 1))


def test_two_thresholds_with_a_lower_include_cutoff() -> None:
    row = two_threshold_row([Q1, Q2], [F1, F2], gate=0.68, include=0.60)
    # Q1 passes the gate (0.80); chunks >= 0.60 are 0.80, 0.70, 0.60 -> pages 5, 2, 5 -> 2 of 3 relevant
    # Q2 is blocked (best 0.66 < 0.68). F1 passes (0.70) and would send 0.70, 0.62 = 2 chunks. F2 blocked.
    assert (row.wrongly_blocked, row.wrongly_passed) == (1, 1)
    assert row.chunks_sent == 3.0
    assert row.include_precision == pytest.approx(2 / 3)
    assert row.include_recall == pytest.approx(0.5)  # (Q1: page 5 found = 1, Q2 blocked = 0) / 2
    assert row.noise_chunks == 2.0


def test_two_thresholds_equal_to_the_single_threshold() -> None:
    row = two_threshold_row([Q1, Q2], [F1, F2], gate=0.68, include=0.68)
    # Q1 sends 0.80, 0.70 -> pages 5, 2 -> 1 of 2 relevant. F1 sends only 0.70 -> 1 chunk.
    assert (row.wrongly_blocked, row.wrongly_passed) == (1, 1)
    assert row.chunks_sent == 2.0 and row.include_precision == pytest.approx(1 / 2)
    assert row.include_recall == pytest.approx(0.5) and row.noise_chunks == 1.0


def test_two_thresholds_when_nothing_is_blocked() -> None:
    row = two_threshold_row([Q1, Q2], [F1, F2], gate=0.60, include=0.60)
    # Q1 sends 3 chunks (precision 2/3, recall 1); Q2 sends 0.66 and 0.64 (precision 1, both pages found)
    assert row.wrongly_blocked == 0 and row.wrongly_passed == 1  # F2 (0.50) still blocked
    assert row.chunks_sent == pytest.approx(2.5)  # (3 + 2) / 2
    assert row.include_precision == pytest.approx(5 / 6)  # (2/3 + 1) / 2
    assert row.include_recall == 1.0
    assert row.noise_chunks == 2.0  # F1 sends 0.70 and 0.62


def test_a_best_score_exactly_at_the_gate_passes_it() -> None:
    row = two_threshold_row([Q2], [], gate=0.66, include=0.64)  # Q2's best score is 0.66
    assert row.wrongly_blocked == 0 and row.chunks_sent == 2.0  # 0.66 and 0.64 both reach include


def test_two_thresholds_when_everything_is_blocked() -> None:
    row = two_threshold_row([Q1, Q2], [F1, F2], gate=0.99, include=0.90)
    assert (row.wrongly_blocked, row.wrongly_passed) == (2, 0)
    assert row.chunks_sent is None and row.include_precision is None and row.noise_chunks is None
    assert row.include_recall == 0.0


def test_two_thresholds_validate_their_inputs() -> None:
    with pytest.raises(ValueError):
        two_threshold_row([Q1], [F1], gate=0.6, include=0.7)
    assert two_threshold_row([([], {5})], [], gate=0.5, include=0.5).wrongly_blocked == 1  # no hits = blocked


# ---------------------------------------------------------------------- outcomes


@pytest.mark.parametrize(
    ("expect_fallback", "used_fallback", "correctness", "is_refusal", "expected"),
    [
        (False, True, None, None, Outcome.WRONGLY_BLOCKED),
        (False, True, "correct", False, Outcome.WRONGLY_BLOCKED),  # a fallback is never judged "correct"
        (False, False, "correct", False, Outcome.CORRECT),
        (False, False, "partially_correct", False, Outcome.PARTIAL),
        (False, False, "incorrect", False, Outcome.INCORRECT),
        (False, False, "incorrect", True, Outcome.INCORRECT),  # refused although the answer exists
        (False, False, None, None, Outcome.UNJUDGED),
        (True, True, None, None, Outcome.CORRECT_FALLBACK),
        (True, False, "correct", True, Outcome.CORRECT_REFUSAL),
        (True, False, "correct", False, Outcome.WRONGLY_ANSWERED),
        (True, False, "partially_correct", True, Outcome.WRONGLY_ANSWERED),
        (True, False, "incorrect", False, Outcome.WRONGLY_ANSWERED),
        (True, False, None, None, Outcome.UNJUDGED),
    ],
)
def test_classify_outcome(expect_fallback, used_fallback, correctness, is_refusal, expected) -> None:
    assert classify_outcome(expect_fallback, used_fallback, correctness, is_refusal) is expected


@pytest.mark.parametrize("expect_fallback", [False, True])
def test_classify_outcome_rejects_an_unknown_label(expect_fallback: bool) -> None:
    with pytest.raises(ValueError):
        classify_outcome(expect_fallback, False, "mostly_ok", False)
