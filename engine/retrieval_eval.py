import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


class RetrievalEvaluator:
    """
    Retrieval-focused evaluator for Role 1:
    - HitRate@K (K configurable, default 1/3/5)
    - MRR
    - Top-K sensitivity report
    - Optional retrieval-quality vs answer-quality correlation
    """

    def __init__(self, top_ks: Sequence[int] = (1, 3, 5)) -> None:
        cleaned = sorted({k for k in top_ks if isinstance(k, int) and k > 0})
        self.top_ks = tuple(cleaned or [1, 3, 5])

    @staticmethod
    def _as_str_list(values: Optional[Iterable[Any]]) -> List[str]:
        if values is None:
            return []
        return [str(v) for v in values if v is not None and str(v).strip()]

    @staticmethod
    def _get_by_path(record: Dict[str, Any], dotted_path: str) -> Optional[Any]:
        current: Any = record
        for key in dotted_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    @staticmethod
    def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
        if len(xs) < 2 or len(xs) != len(ys):
            return None
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x == 0 or var_y == 0:
            return None
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        return cov / math.sqrt(var_x * var_y)

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> float:
        """
        HitRate@K = 1 nếu ít nhất một expected_id nằm trong top-k retrieved_ids, ngược lại = 0.
        """
        normalized_expected = set(self._as_str_list(expected_ids))
        normalized_retrieved = self._as_str_list(retrieved_ids)

        if not normalized_expected or top_k <= 0:
            return 0.0

        top_retrieved = normalized_retrieved[:top_k]
        hit = any(doc_id in normalized_expected for doc_id in top_retrieved)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR (single-query): Reciprocal rank của tài liệu đúng đầu tiên trong retrieved list.
        """
        normalized_expected = set(self._as_str_list(expected_ids))
        normalized_retrieved = self._as_str_list(retrieved_ids)

        if not normalized_expected:
            return 0.0

        for idx, doc_id in enumerate(normalized_retrieved, start=1):
            if doc_id in normalized_expected:
                return 1.0 / idx
        return 0.0

    def evaluate_case(
        self,
        case_id: str,
        expected_ids: List[str],
        retrieved_ids: List[str],
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "case_id": case_id,
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "expected_retrieval_ids": self._as_str_list(expected_ids),
            "retrieved_ids": self._as_str_list(retrieved_ids),
        }
        for k in self.top_ks:
            metrics[f"hit_rate@{k}"] = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=k)
        return metrics

    async def evaluate_batch(
        self,
        dataset: List[Dict[str, Any]],
        answer_score_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dataset input cho mỗi case:
        - expected_retrieval_ids: List[str]
        - retrieved_ids: List[str]
        - optional score field cho correlation (vd: "judge.final_score")
        """
        case_metrics: List[Dict[str, Any]] = []
        skipped_cases = 0

        mrr_values: List[float] = []
        hit_values_by_k: Dict[int, List[float]] = {k: [] for k in self.top_ks}
        retrieval_for_corr: List[float] = []
        answer_for_corr: List[float] = []

        for idx, case in enumerate(dataset):
            expected_ids = self._as_str_list(case.get("expected_retrieval_ids"))
            retrieved_ids = self._as_str_list(case.get("retrieved_ids"))
            case_id = str(case.get("case_id") or f"case_{idx + 1}")

            if not expected_ids or not retrieved_ids:
                skipped_cases += 1
                continue

            per_case = self.evaluate_case(case_id, expected_ids, retrieved_ids)
            case_metrics.append(per_case)
            mrr_values.append(per_case["mrr"])
            for k in self.top_ks:
                hit_values_by_k[k].append(per_case[f"hit_rate@{k}"])

            if answer_score_path:
                answer_score = self._get_by_path(case, answer_score_path)
                if isinstance(answer_score, (int, float)):
                    retrieval_for_corr.append(per_case["mrr"])
                    answer_for_corr.append(float(answer_score))

        evaluated_cases = len(case_metrics)
        avg_mrr = sum(mrr_values) / evaluated_cases if evaluated_cases else 0.0

        summary: Dict[str, Any] = {
            "total_cases": len(dataset),
            "evaluated_cases": evaluated_cases,
            "skipped_cases": skipped_cases,
            "avg_mrr": avg_mrr,
            "per_case": case_metrics,
        }

        for k in self.top_ks:
            avg_hit = sum(hit_values_by_k[k]) / evaluated_cases if evaluated_cases else 0.0
            summary[f"avg_hit_rate@{k}"] = avg_hit
            summary[f"hit_rate@{k}"] = avg_hit

        # Backward compatibility for existing checkers expecting "avg_hit_rate"
        summary["avg_hit_rate"] = summary.get("avg_hit_rate@3", summary.get("avg_hit_rate@1", 0.0))

        summary["top_k_sensitivity"] = {
            f"@{k}": summary[f"avg_hit_rate@{k}"] for k in self.top_ks
        }

        corr = self._pearson(retrieval_for_corr, answer_for_corr)
        if corr is not None:
            summary["retrieval_answer_correlation"] = corr
            if corr >= 0.5:
                relation = "strong_positive"
            elif corr >= 0.2:
                relation = "moderate_positive"
            elif corr > -0.2:
                relation = "weak_or_none"
            else:
                relation = "negative"
            summary["retrieval_answer_relationship"] = relation

        return summary
