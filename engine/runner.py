import asyncio
import time
from typing import Any, Dict, List, Optional

from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    """
    Async benchmark runner with:
      - asyncio.Semaphore  → cap concurrent LLM calls (avoid rate-limit)
      - per-case timeout   → no hung requests
      - exponential-backoff retry  → handle transient API errors
      - latency / cost / token tracking per case
    """

    def __init__(
        self,
        agent,
        evaluator,
        judge,
        semaphore_limit: int = 10,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.semaphore_limit = semaphore_limit
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._retrieval_eval = RetrievalEvaluator(top_ks=(1, 3, 5))

    # ── Single-case execution ─────────────────────────────────────────────────

    async def _execute_single(self, test_case: Dict) -> Dict:
        start = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start

        retrieved_ids = response.get("retrieved_ids", [])
        expected_ids  = test_case.get("expected_retrieval_ids", [])

        retrieval_metrics = self._retrieval_eval.evaluate_case(
            case_id=test_case.get("case_id", ""),
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
        )

        ragas_scores = await self.evaluator.score(test_case, response)
        ragas_scores["retrieval"] = {
            "hit_rate":   retrieval_metrics.get("hit_rate@3", 0.0),
            "hit_rate@1": retrieval_metrics.get("hit_rate@1", 0.0),
            "hit_rate@3": retrieval_metrics.get("hit_rate@3", 0.0),
            "hit_rate@5": retrieval_metrics.get("hit_rate@5", 0.0),
            "mrr":        retrieval_metrics.get("mrr", 0.0),
        }

        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
            case_id=test_case.get("case_id"),
        )

        return {
            "case_id":                test_case.get("case_id", ""),
            "case_type":              test_case.get("case_type", ""),
            "difficulty":             test_case.get("difficulty", ""),
            "question":               test_case["question"],
            "expected_answer":        test_case["expected_answer"],
            "agent_response":         response["answer"],
            "expected_retrieval_ids": expected_ids,
            "retrieved_ids":          retrieved_ids,
            "latency_s":              round(latency, 4),
            "ragas":                  ragas_scores,
            "judge":                  judge_result,
            "status":                 "pass" if judge_result["final_score"] >= 3 else "fail",
            "metadata":               response.get("metadata", {}),
            "error":                  None,
        }

    def _error_result(self, test_case: Dict, error: str) -> Dict:
        return {
            "case_id":                test_case.get("case_id", ""),
            "case_type":              test_case.get("case_type", ""),
            "difficulty":             test_case.get("difficulty", ""),
            "question":               test_case["question"],
            "expected_answer":        test_case["expected_answer"],
            "agent_response":         "",
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "retrieved_ids":          [],
            "latency_s":              -1.0,
            "ragas":                  {},
            "judge": {
                "final_score": 0, "agreement_rate": 0.0,
                "individual_scores": {}, "cost_usd": 0.0,
                "tokens_in": 0, "tokens_out": 0,
            },
            "status":   "error",
            "metadata": {},
            "error":    error,
        }

    # ── Retry + semaphore wrapper ─────────────────────────────────────────────

    async def _run_with_retry(
        self, test_case: Dict, semaphore: asyncio.Semaphore
    ) -> Dict:
        async with semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    return await asyncio.wait_for(
                        self._execute_single(test_case),
                        timeout=self.timeout_s,
                    )
                except asyncio.TimeoutError:
                    if attempt == self.max_retries:
                        return self._error_result(test_case, "timeout")
                except Exception as exc:
                    if attempt == self.max_retries:
                        return self._error_result(test_case, str(exc))
                await asyncio.sleep(2 ** attempt)
        return self._error_result(test_case, "semaphore_exhausted")

    # ── Public interface ──────────────────────────────────────────────────────

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Run all cases concurrently, capped by semaphore_limit."""
        semaphore = asyncio.Semaphore(self.semaphore_limit)
        tasks = [self._run_with_retry(case, semaphore) for case in dataset]
        return list(await asyncio.gather(*tasks))

    # ── Summary statistics ────────────────────────────────────────────────────

    @staticmethod
    def compute_summary_stats(results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        valid = [r for r in results if r.get("status") != "error"]
        error_count = total - len(valid)

        if not valid:
            return {"total": total, "error_count": error_count, "avg_score": 0.0}

        latencies = sorted(r["latency_s"] for r in valid if r["latency_s"] >= 0)
        p95_idx   = max(0, int(len(latencies) * 0.95) - 1)
        p95_lat   = latencies[p95_idx]   if latencies else 0.0
        avg_lat   = sum(latencies) / len(latencies) if latencies else 0.0

        avg_score      = sum(r["judge"]["final_score"]   for r in valid) / len(valid)
        agreement_rate = sum(r["judge"].get("agreement_rate", 0) for r in valid) / len(valid)

        hit3 = sum(
            r["ragas"].get("retrieval", {}).get("hit_rate@3",
                r["ragas"].get("retrieval", {}).get("hit_rate", 0))
            for r in valid
        ) / len(valid)

        total_cost       = sum(r["judge"].get("cost_usd",    0) for r in results)
        total_tokens_in  = sum(r["judge"].get("tokens_in",   0) for r in results)
        total_tokens_out = sum(r["judge"].get("tokens_out",  0) for r in results)

        pass_count  = sum(1 for r in results if r.get("status") == "pass")
        fail_count  = sum(1 for r in results if r.get("status") == "fail")

        return {
            "avg_score":       round(avg_score,      4),
            "hit_rate":        round(hit3,            4),  # alias expected by check_lab.py
            "hit_rate@3":      round(hit3,            4),
            "agreement_rate":  round(agreement_rate,  4),
            "p95_latency_s":   round(p95_lat,         4),
            "avg_latency_s":   round(avg_lat,         4),
            "total_cost_usd":  round(total_cost,      6),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out":total_tokens_out,
            "total":           total,
            "pass_count":      pass_count,
            "fail_count":      fail_count,
            "error_count":     error_count,
        }

    # ── Failure clustering ────────────────────────────────────────────────────

    @staticmethod
    def cluster_failures(results: List[Dict]) -> Dict[str, List[str]]:
        """Group failed cases by taxonomy for failure_analysis.md."""
        clusters: Dict[str, List[str]] = {
            "prompt_attack":       [],
            "hallucination":       [],
            "retrieval_miss":      [],
            "incomplete":          [],
            "tone_mismatch":       [],
        }
        for r in results:
            if r.get("status") != "fail":
                continue
            cid  = r.get("case_id", "?")
            ctype = r.get("case_type", "")
            hr3  = r.get("ragas", {}).get("retrieval", {}).get("hit_rate@3", 1.0)
            score = r["judge"].get("final_score", 5)

            if "adversarial" in ctype:
                clusters["prompt_attack"].append(cid)
            elif hr3 == 0.0 and score <= 2:
                clusters["retrieval_miss"].append(cid)
            elif score <= 2:
                clusters["hallucination"].append(cid)
            elif "multi_turn" in ctype:
                clusters["incomplete"].append(cid)
            else:
                clusters["tone_mismatch"].append(cid)
        return clusters
