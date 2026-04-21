import asyncio
import hashlib
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional


class LLMJudge:
    """
    Multi-judge LLM evaluator.

    Consensus rule:
      - |score_a - score_b| <= 1  →  average (agreement)
      - |score_a - score_b| >  1  →  call tiebreaker model (conflict)

    Position-bias check: judge_a is re-run with answer/ground_truth order swapped;
    the delta between the two scores measures how sensitive the judge is to position.

    Mock mode: activated automatically when OPENAI_API_KEY is not set.
    Scores are deterministic (hash-based) so tests are reproducible.
    """

    PRICE_PER_1M: Dict[str, Dict[str, float]] = {
        "gpt-4o":       {"input": 5.00, "output": 15.00},
        "gpt-4o-mini":  {"input": 0.15, "output":  0.60},
    }

    def __init__(
        self,
        model_a: str = "gpt-4o-mini",
        model_b: str = "gpt-4o-mini",
        tiebreaker_model: str = "gpt-4o",
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.tiebreaker_model = tiebreaker_model
        self._client = None
        self._mock_mode = not bool(os.getenv("OPENAI_API_KEY"))
        if self._mock_mode:
            print("[LLMJudge] OPENAI_API_KEY not set - running in mock mode.")

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _prompt(question: str, answer: str, ground_truth: str) -> str:
        return (
            "Bạn là chuyên gia đánh giá chất lượng câu trả lời AI. "
            "Chấm điểm câu trả lời từ 1 đến 5.\n\n"
            f"Câu hỏi: {question}\n"
            f"Câu trả lời cần đánh giá: {answer}\n"
            f"Câu trả lời tham chiếu (ground truth): {ground_truth}\n\n"
            "Thang điểm:\n"
            "5 - Hoàn toàn chính xác, đầy đủ, phù hợp context\n"
            "4 - Phần lớn đúng, thiếu chi tiết nhỏ\n"
            "3 - Đúng một phần, thiếu thông tin quan trọng\n"
            "2 - Phần lớn sai hoặc không liên quan\n"
            "1 - Hoàn toàn sai, bịa đặt hoặc vi phạm policy\n\n"
            "Chỉ trả về MỘT chữ số từ 1 đến 5. Không giải thích thêm."
        )

    @staticmethod
    def _prompt_swapped(question: str, answer: str, ground_truth: str) -> str:
        """Swapped order to detect position bias."""
        return (
            "Bạn là chuyên gia đánh giá chất lượng câu trả lời AI. "
            "Chấm điểm câu trả lời từ 1 đến 5.\n\n"
            f"Câu hỏi: {question}\n"
            f"Câu trả lời tham chiếu (ground truth): {ground_truth}\n"
            f"Câu trả lời cần đánh giá: {answer}\n\n"
            "Thang điểm:\n"
            "5 - Hoàn toàn chính xác, đầy đủ, phù hợp context\n"
            "4 - Phần lớn đúng, thiếu chi tiết nhỏ\n"
            "3 - Đúng một phần, thiếu thông tin quan trọng\n"
            "2 - Phần lớn sai hoặc không liên quan\n"
            "1 - Hoàn toàn sai, bịa đặt hoặc vi phạm policy\n\n"
            "Chỉ trả về MỘT chữ số từ 1 đến 5. Không giải thích thêm."
        )

    # ── Cost helper ───────────────────────────────────────────────────────────

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        prices = self.PRICE_PER_1M.get(model, {"input": 5.0, "output": 15.0})
        return (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000

    # ── API / mock call ───────────────────────────────────────────────────────

    async def _call_api(self, model: str, prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        match = re.search(r"[1-5]", raw)
        score = int(match.group()) if match else 3
        return {
            "score": score,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": self._cost(model, tokens_in, tokens_out),
        }

    @staticmethod
    def _mock_call(salt: str) -> Dict[str, Any]:
        """Deterministic mock score based on hash so runs are reproducible."""
        h = int(hashlib.md5(salt.encode()).hexdigest(), 16)
        score = (h % 4) + 2  # 2, 3, 4, or 5 — realistic spread
        return {"score": score, "tokens_in": 120, "tokens_out": 2, "cost_usd": 0.0}

    async def _judge(self, model: str, prompt: str, salt: str = "") -> Dict[str, Any]:
        if self._mock_mode:
            return self._mock_call(salt or prompt[:64])
        return await self._call_api(model, prompt)

    # ── Core evaluation ───────────────────────────────────────────────────────

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt_normal  = self._prompt(question, answer, ground_truth)
        prompt_swapped = self._prompt_swapped(question, answer, ground_truth)

        # Judge A and Judge B run in parallel
        result_a, result_b = await asyncio.gather(
            self._judge(self.model_a, prompt_normal,  salt=f"A:{case_id}:{question}"),
            self._judge(self.model_b, prompt_normal,  salt=f"B:{case_id}:{question}"),
        )

        score_a, score_b = result_a["score"], result_b["score"]
        diff = abs(score_a - score_b)

        # Conflict handling
        tiebreaker: Optional[Dict] = None
        if diff > 1:
            tiebreaker = await self._judge(
                self.tiebreaker_model, prompt_normal,
                salt=f"TB:{case_id}:{question}",
            )
            final_score = float(tiebreaker["score"])
        else:
            final_score = (score_a + score_b) / 2.0

        # Position-bias check: re-run judge_a with swapped prompt
        bias_result = await self._judge(
            self.model_a, prompt_swapped,
            salt=f"BIAS:{case_id}:{question}",
        )
        position_bias_delta = abs(score_a - bias_result["score"])

        # Aggregate token / cost
        total_tokens_in  = result_a["tokens_in"]  + result_b["tokens_in"]  + bias_result["tokens_in"]
        total_tokens_out = result_a["tokens_out"] + result_b["tokens_out"] + bias_result["tokens_out"]
        total_cost       = result_a["cost_usd"]   + result_b["cost_usd"]   + bias_result["cost_usd"]
        if tiebreaker:
            total_tokens_in  += tiebreaker["tokens_in"]
            total_tokens_out += tiebreaker["tokens_out"]
            total_cost       += tiebreaker["cost_usd"]

        agreement = diff <= 1

        return {
            "case_id":             case_id,
            "final_score":         final_score,
            "score_a":             score_a,
            "score_b":             score_b,
            "conflict":            not agreement,
            "agreement":           agreement,
            "tiebreaker_score":    tiebreaker["score"] if tiebreaker else None,
            "position_bias_delta": position_bias_delta,
            "individual_scores":   {self.model_a: score_a, self.model_b: score_b},
            "tokens_in":           total_tokens_in,
            "tokens_out":          total_tokens_out,
            "cost_usd":            total_cost,
        }

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runner-compatible interface: returns flat dict with final_score + agreement_rate."""
        r = await self.evaluate_single(question, answer, ground_truth, case_id)
        return {
            "final_score":         r["final_score"],
            "agreement_rate":      1.0 if r["agreement"] else 0.0,
            "individual_scores":   r["individual_scores"],
            "conflict":            r["conflict"],
            "tiebreaker_score":    r["tiebreaker_score"],
            "position_bias_delta": r["position_bias_delta"],
            "tokens_in":           r["tokens_in"],
            "tokens_out":          r["tokens_out"],
            "cost_usd":            r["cost_usd"],
        }

    # ── Batch-level statistics ────────────────────────────────────────────────

    @staticmethod
    def cohen_kappa(scores_a: List[int], scores_b: List[int]) -> float:
        """
        Cohen's Kappa for two integer-score raters (1–5 scale).
        Measures inter-rater agreement beyond what is expected by chance.
        κ = (Po - Pe) / (1 - Pe)
        """
        n = len(scores_a)
        if n == 0:
            return 0.0
        po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
        count_a = Counter(scores_a)
        count_b = Counter(scores_b)
        pe = sum(count_a[k] * count_b[k] for k in range(1, 6)) / (n * n)
        if 1 - pe == 0:
            return 1.0
        return round((po - pe) / (1 - pe), 4)

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        """Legacy stub — position bias is measured per-case inside evaluate_single."""
        return {"note": "Position bias checked per-case via evaluate_single (prompt swap)."}
