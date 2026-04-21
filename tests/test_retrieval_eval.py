import asyncio

import pytest

from engine.retrieval_eval import RetrievalEvaluator


@pytest.mark.parametrize(
    "expected_ids,retrieved_ids,top_k,expected",
    [
        (["doc_a"], ["doc_a", "doc_b"], 1, 1.0),
        (["doc_a"], ["doc_b", "doc_a"], 1, 0.0),
        (["doc_a"], ["doc_b", "doc_a"], 2, 1.0),
        (["doc_a", "doc_c"], ["doc_x", "doc_c"], 2, 1.0),
        (["doc_a"], [], 3, 0.0),
    ],
)
def test_calculate_hit_rate(expected_ids, retrieved_ids, top_k, expected):
    evaluator = RetrievalEvaluator()
    assert evaluator.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k) == expected


@pytest.mark.parametrize(
    "expected_ids,retrieved_ids,expected_mrr",
    [
        (["doc_a"], ["doc_a", "doc_b"], 1.0),
        (["doc_a"], ["doc_b", "doc_a"], 0.5),
        (["doc_a", "doc_c"], ["doc_x", "doc_c"], 0.5),
        (["doc_a"], ["doc_x", "doc_y"], 0.0),
    ],
)
def test_calculate_mrr(expected_ids, retrieved_ids, expected_mrr):
    evaluator = RetrievalEvaluator()
    assert evaluator.calculate_mrr(expected_ids, retrieved_ids) == expected_mrr


def test_evaluate_batch_returns_expected_aggregates():
    evaluator = RetrievalEvaluator(top_ks=(1, 3, 5))
    dataset = [
        {
            "case_id": "c1",
            "expected_retrieval_ids": ["doc_a"],
            "retrieved_ids": ["doc_a", "doc_b"],
            "judge": {"final_score": 4.5},
        },
        {
            "case_id": "c2",
            "expected_retrieval_ids": ["doc_c"],
            "retrieved_ids": ["doc_b", "doc_c"],
            "judge": {"final_score": 3.0},
        },
        {
            "case_id": "c3",
            "expected_retrieval_ids": ["doc_x"],
            "retrieved_ids": ["doc_b", "doc_c"],
            "judge": {"final_score": 1.0},
        },
    ]

    result = asyncio.run(evaluator.evaluate_batch(dataset, answer_score_path="judge.final_score"))

    assert result["total_cases"] == 3
    assert result["evaluated_cases"] == 3
    assert result["skipped_cases"] == 0
    assert result["avg_hit_rate@1"] == pytest.approx(1 / 3)
    assert result["avg_hit_rate@3"] == pytest.approx(2 / 3)
    assert result["avg_hit_rate@5"] == pytest.approx(2 / 3)
    assert result["avg_mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert "@1" in result["top_k_sensitivity"]
    assert "@3" in result["top_k_sensitivity"]
    assert "retrieval_answer_correlation" in result
