import asyncio
from typing import Dict, List


class MainAgent:
    """
    Simulated RAG agent that supports V1 (baseline) and V2 (improved) modes.

    V1: simple keyword-based retrieval, returns only first context chunk.
    V2: broader retrieval fallback + returns all matched context chunks,
        producing more complete answers.

    To use a real LLM, replace _retrieve() and _generate() with actual
    vector-DB lookup and LLM completion calls.
    """

    # Simulated knowledge base (mirrors the doc IDs in synthetic_gen.py)
    KNOWLEDGE_BASE: Dict[str, str] = {
        "doc_retrieval_metrics":    "Hit Rate đo có tìm thấy tài liệu đúng trong top-k hay không, còn MRR phạt theo vị trí tài liệu đúng đầu tiên.",
        "doc_multi_judge_consensus":"Multi-Judge giúp giảm thiên lệch của một model đơn lẻ và tăng độ tin cậy qua đồng thuận liên model.",
        "doc_cohen_kappa":          "Cohen's Kappa đo mức độ đồng thuận vượt trên ngẫu nhiên giữa các judge.",
        "doc_position_bias":        "Position Bias là hiện tượng judge ưu ái phương án xuất hiện trước, dù chất lượng tương đương.",
        "doc_async_runner":         "Async runner cho phép chạy nhiều test case song song, giảm đáng kể tổng thời gian benchmark.",
        "doc_cost_tracking":        "Cần theo dõi token vào/ra và chi phí ước tính theo từng test case và toàn pipeline.",
        "doc_release_gate":         "Release Gate thường dùng delta chất lượng, độ ổn định retrieval, agreement rate và hiệu năng.",
        "doc_chunking_strategy":    "Chunking kém có thể làm rơi mất ngữ cảnh quan trọng, khiến model suy luận sai và hallucinate.",
        "doc_reranking":            "Reranking cải thiện thứ hạng tài liệu liên quan, giúp tăng Hit@K và giảm lỗi trả lời sai ngữ cảnh.",
        "doc_prompt_guardrails":    "Guardrail buộc model bám context, từ chối yêu cầu trái chính sách và giảm prompt injection.",
        "doc_scope_policy":         "Agent cần nêu rõ không có thông tin trong ngữ cảnh hiện có thay vì tự bịa câu trả lời.",
        "doc_clarification_policy": "Agent nên hỏi làm rõ yêu cầu trước khi đưa ra kết luận để tránh trả lời sai phạm vi.",
        "doc_sla_v2":               "Bản chính sách mới quy định SLA phản hồi chuẩn là 2 giờ cho ticket ưu tiên cao.",
        "doc_sla_legacy":           "Tài liệu legacy ghi SLA phản hồi là 4 giờ, đã bị thay thế bởi bản mới.",
        "doc_incident_playbook":    "Bước đầu tiên là xác nhận phạm vi ảnh hưởng và kích hoạt kênh incident bridge.",
    }

    KEYWORD_MAP: Dict[str, List[str]] = {
        "hit rate":         ["doc_retrieval_metrics"],
        "mrr":              ["doc_retrieval_metrics"],
        "retrieval":        ["doc_retrieval_metrics", "doc_chunking_strategy"],
        "multi-judge":      ["doc_multi_judge_consensus"],
        "multi judge":      ["doc_multi_judge_consensus"],
        "judge":            ["doc_multi_judge_consensus", "doc_position_bias"],
        "kappa":            ["doc_cohen_kappa"],
        "đồng thuận":       ["doc_cohen_kappa", "doc_multi_judge_consensus"],
        "position bias":    ["doc_position_bias"],
        "thiên lệch":       ["doc_position_bias"],
        "async":            ["doc_async_runner"],
        "song song":        ["doc_async_runner"],
        "cost":             ["doc_cost_tracking"],
        "chi phí":          ["doc_cost_tracking"],
        "token":            ["doc_cost_tracking"],
        "release gate":     ["doc_release_gate"],
        "release":          ["doc_release_gate"],
        "chunk":            ["doc_chunking_strategy"],
        "hallucination":    ["doc_chunking_strategy", "doc_prompt_guardrails"],
        "rerank":           ["doc_reranking"],
        "guardrail":        ["doc_prompt_guardrails"],
        "injection":        ["doc_prompt_guardrails"],
        "bỏ qua":           ["doc_prompt_guardrails"],
        "ignore":           ["doc_prompt_guardrails"],
        "sla":              ["doc_sla_v2", "doc_sla_legacy"],
        "incident":         ["doc_incident_playbook"],
        "mơ hồ":            ["doc_clarification_policy"],
        "ambiguous":        ["doc_clarification_policy"],
        "làm rõ":           ["doc_clarification_policy"],
    }

    def __init__(self, version: str = "v1") -> None:
        self.version = version
        self.name = f"SupportAgent-{version}"

    def _retrieve(self, question: str) -> List[str]:
        q = question.lower()
        retrieved: List[str] = []
        for keyword, doc_ids in self.KEYWORD_MAP.items():
            if keyword in q:
                for doc_id in doc_ids:
                    if doc_id not in retrieved:
                        retrieved.append(doc_id)

        # V2 has a broader fallback — if nothing matched, return the scope policy doc
        if self.version == "v2" and not retrieved:
            retrieved = ["doc_scope_policy"]

        return retrieved[:5]  # cap at top-5 like a real retriever

    def _generate(self, question: str, contexts: List[str]) -> str:
        q = question.lower()

        # Reject prompt-injection attempts
        injection_signals = ["bỏ qua", "ignore previous", "ignore all", "đừng dựa", "cứ trả lời"]
        if any(sig in q for sig in injection_signals):
            ctx_answer = contexts[0] if contexts else ""
            return (
                "Yêu cầu bỏ qua context bị từ chối. "
                f"Thông tin đúng theo tài liệu: {ctx_answer}"
            )

        if not contexts:
            # Out-of-scope or ambiguous
            ambiguous_signals = ["chỉ số này", "làm vậy", "case này", "điểm judge"]
            if any(sig in q for sig in ambiguous_signals):
                return (
                    "Câu hỏi còn mơ hồ. Vui lòng làm rõ metric hoặc ngưỡng bạn muốn "
                    "đánh giá trước khi kết luận."
                )
            return (
                "Không có thông tin liên quan trong tài liệu benchmark hiện tại. "
                "Vui lòng cung cấp ngữ cảnh phù hợp với phạm vi hệ thống."
            )

        # V2: combine all context chunks for a richer answer
        # V1: only use the first chunk
        if self.version == "v2":
            return " ".join(contexts)
        return contexts[0]

    async def query(self, question: str) -> Dict:
        # Simulated latency — 0.05 s per call keeps the 60-case benchmark fast
        await asyncio.sleep(0.05)

        retrieved_ids = self._retrieve(question)
        contexts = [
            self.KNOWLEDGE_BASE[doc_id]
            for doc_id in retrieved_ids
            if doc_id in self.KNOWLEDGE_BASE
        ]
        answer = self._generate(question, contexts)

        return {
            "answer":       answer,
            "retrieved_ids": retrieved_ids,
            "contexts":     contexts,
            "metadata": {
                "model":               f"simulated-rag-{self.version}",
                "version":             self.version,
                "tokens_in":           len(question.split()) * 2,
                "tokens_out":          len(answer.split()),
                "estimated_cost_usd":  0.0,
            },
        }


if __name__ == "__main__":
    async def _smoke_test():
        for ver in ("v1", "v2"):
            agent = MainAgent(version=ver)
            resp = await agent.query("Async runner giúp pipeline benchmark cải thiện như thế nào?")
            print(f"[{ver}] answer: {resp['answer'][:80]}...")
            print(f"[{ver}] retrieved_ids: {resp['retrieved_ids']}")
    asyncio.run(_smoke_test())
