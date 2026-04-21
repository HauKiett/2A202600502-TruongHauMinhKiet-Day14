from collections import Counter

from data.synthetic_gen import (
    CASE_DISTRIBUTION,
    build_golden_dataset,
    build_hard_cases_pack,
    validate_dataset_schema,
)


def test_build_golden_dataset_distribution_and_schema():
    cases = build_golden_dataset(seed=42)
    validate_dataset_schema(cases)

    distribution = Counter(case["case_type"] for case in cases)
    assert len(cases) == 60
    assert distribution == Counter(CASE_DISTRIBUTION)

    sample = cases[0]
    assert "case_id" in sample
    assert "expected_retrieval_ids" in sample
    assert isinstance(sample["expected_retrieval_ids"], list)
    assert isinstance(sample["tags"], list)


def test_hard_cases_pack_meets_bonus_target():
    cases = build_golden_dataset(seed=42)
    hard_pack = build_hard_cases_pack(cases)
    summary = hard_pack["summary"]

    assert summary["hard_case_total"] == 30
    assert sorted(summary["case_types_covered"]) == sorted(
        [
            "adversarial_prompt_injection",
            "conflicting_information",
            "edge_ambiguous_ooc",
            "multi_turn",
        ]
    )
    assert "red-team coverage" in summary["notes"]
