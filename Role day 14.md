# Kế Hoạch 2 Người Đạt 100% Task + Bonus (Lab 14 AI Evaluation)

## Tóm tắt
- Mục tiêu: khóa đủ điều kiện **60/60 điểm nhóm** và tối đa hóa điểm cá nhân qua ownership rõ ràng + commit evidence.
- Chia theo 2 trục độc lập để làm song song: **Role 1 (Data/Retrieval/Analysis)** và **Role 2 (Judge/Runner/Regression)**.
- Chốt trước contract dữ liệu/metrics để khi merge không phát sinh quyết định mới.

## Thay đổi triển khai (decision-complete)

### 1) Contract interface bắt buộc (chốt ngay từ đầu)
- `data/golden_set.jsonl` mỗi dòng có: `case_id`, `question`, `expected_answer`, `expected_retrieval_ids` (list), `difficulty`, `case_type`, `tags`.
- `agent.query()` trả về: `answer`, `retrieved_ids` (ranked list), `contexts`, `metadata.tokens_in`, `metadata.tokens_out`, `metadata.model`, `metadata.estimated_cost_usd`.
- `reports/benchmark_results.json` chứa per-case: retrieval metrics, judge metrics, latency, cost, pass/fail, error_cluster.
- `reports/summary.json` chứa: `avg_score`, `hit_rate@3`, `mrr`, `agreement_rate`, `cohen_kappa`, `p95_latency_s`, `total_cost_usd`, `regression.delta_*`, `release_gate.decision`.

### 2) Role 1: Data/Retrieval/Analysis Lead
- Ownership chính: [data/synthetic_gen.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/data/synthetic_gen.py), [engine/retrieval_eval.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/engine/retrieval_eval.py), [analysis/failure_analysis.md](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/analysis/failure_analysis.md).
- Sinh dataset **60 cases**: 30 normal, 12 adversarial/prompt injection, 8 out-of-context/ambiguous, 6 conflicting-info, 4 multi-turn; tất cả có ground-truth IDs.
- Implement retrieval eval thật: `HitRate@1/@3/@5`, `MRR`, batch aggregate, mapping retrieval-quality ↔ answer-quality.
- Chuẩn hóa failure clustering (Hallucination, Incomplete, Tone mismatch, Retrieval miss, Prompt attack) và điền 5 Whys cho 3 case tệ nhất.
- Bonus trong scope Role 1: top-k sensitivity report + “hard-cases pack” có tỷ lệ fail ban đầu >20% để chứng minh stress test.

### 3) Role 2: Judge/Runner/Regression Lead
- Ownership chính: [engine/llm_judge.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/engine/llm_judge.py), [engine/runner.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/engine/runner.py), [main.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/main.py), [agent/main_agent.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/agent/main_agent.py), [check_lab.py](d:/Vinuniversity/Day%2014/Lab14-AI-Evaluation-Benchmarking/check_lab.py).
- Multi-judge consensus với 2 model (default: `gpt-4o` + `gpt-4o-mini`); conflict rule: lệch `<=1` lấy trung bình, lệch `>1` gọi tie-breaker.
- Tính `agreement_rate` + `cohen_kappa`; thêm check position-bias bằng hoán đổi thứ tự phương án.
- Nâng async runner: semaphore + retry + timeout + thu thập latency/cost/token, bảo đảm 50 cases < 2 phút.
- Regression V1 vs V2 + auto release gate:
  - Release khi `delta_avg_score >= +0.2`,
  - `hit_rate@3` không giảm quá 0.02,
  - `agreement_rate >= 0.70`,
  - `p95_latency_s <= 1.2 * V1`.
- Bonus trong scope Role 2: judge cascade/caching để giảm **>=30% total eval cost** mà không giảm `avg_score` quá 0.1.

### 4) Lịch thực thi song song (2 người)
1. 30 phút đầu: chốt contract schema + ngưỡng gate + rubric mapping.
2. 120 phút tiếp: Role 1 và Role 2 làm độc lập đúng ownership.
3. 45 phút tiếp: merge + chạy E2E + tuning để đạt ngưỡng performance/cost.
4. 25 phút cuối: hoàn tất `failure_analysis.md`, reflections cá nhân, kiểm tra nộp bài.

## Test plan và tiêu chí đạt max điểm
- Unit test bắt buộc: retrieval metrics, consensus/conflict logic, kappa, release gate decision.
- Integration test: `python data/synthetic_gen.py` → `python main.py` → `python check_lab.py`.
- Performance test: benchmark 50+ cases hoàn thành < 120s.
- Acceptance theo rubric:
  - Retrieval Evaluation: có Hit Rate + MRR + giải thích liên hệ với answer quality.
  - Dataset & SDG: 50+ cases + red teaming thành công.
  - Multi-judge: >=2 judge + agreement + conflict handling.
  - Regression: có delta V1/V2 + release/rollback tự động.
  - Performance: async + cost/token report.
  - Failure Analysis: 5 Whys sâu, chỉ ra root cause hệ thống.
- Evidence điểm cá nhân: mỗi người tối thiểu 6 commit kỹ thuật trong module own + `analysis/reflections/reflection_[Tên].md`.

## Assumptions và default đã chốt
- Dùng OpenAI key sẵn có; nếu thiếu provider thứ 2 thì vẫn dùng 2 model OpenAI khác nhau để đảm bảo tiêu chí “multi-judge”.
- `reports/` và `data/golden_set.jsonl` được generate local theo `.gitignore`; khi nộp cần chạy lại để tạo artifacts.
- Không mở rộng scope ngoài rubric (không làm feature sản phẩm mới), chỉ tối ưu để ăn điểm tối đa + bonus kỹ thuật trong đúng pipeline eval.
