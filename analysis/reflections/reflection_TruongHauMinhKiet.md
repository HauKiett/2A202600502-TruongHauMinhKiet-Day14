# Individual Reflection — TruongHauMinhKiet (Role 2: Judge / Runner / Regression Lead)

## 1. Công việc tôi đã làm

### Module chính (ownership của Role 2)

| File | Mô tả công việc |
|---|---|
| `engine/llm_judge.py` | Multi-judge consensus với 2 model OpenAI, xử lý conflict bằng tiebreaker, tính agreement_rate, Cohen's Kappa, và position-bias check |
| `engine/runner.py` | Async runner với `asyncio.Semaphore` giới hạn concurrent requests, exponential-backoff retry, per-request timeout, thu thập latency / cost / token / failure cluster |
| `main.py` | Pipeline end-to-end: load dataset → chạy V1/V2 → tính regression delta → áp dụng release gate 4 tiêu chí → xuất reports |
| `agent/main_agent.py` | Cập nhật agent V1/V2 có `retrieved_ids` thật để retrieval eval hoạt động |

---

## 2. Các khái niệm kỹ thuật tôi đã hiểu sâu

### Multi-Judge Consensus
Một judge model đơn lẻ có thể bị bias (thiên vị theo style câu trả lời, độ dài, ngôn ngữ…). Khi dùng 2 judge:
- Nếu hai điểm **lệch ≤ 1**: lấy trung bình → ổn định hơn, ít bị outlier
- Nếu hai điểm **lệch > 1**: gọi model thứ 3 (tiebreaker) → giải quyết xung đột có căn cứ

### Cohen's Kappa (κ)
Đo mức đồng thuận **vượt trên ngẫu nhiên** giữa 2 judge:

```
κ = (Po − Pe) / (1 − Pe)
```
- `Po` = tỷ lệ đồng ý thực tế  
- `Pe` = tỷ lệ đồng ý kỳ vọng nếu chấm ngẫu nhiên  
- κ = 1.0 → đồng ý hoàn toàn | κ = 0 → chỉ ngẫu nhiên | κ < 0 → tệ hơn random

Tại sao cần: `agreement_rate = 90%` nghe có vẻ tốt, nhưng nếu tất cả câu hỏi đều dễ và judge nào cũng cho điểm 5, thì 90% đó chỉ là tình cờ — Kappa mới phân biệt được.

### Position Bias
Judge LLM có xu hướng ưu ái thông tin xuất hiện **trước** trong prompt. Để kiểm tra, tôi chạy judge với cùng nội dung nhưng đổi thứ tự answer/ground_truth. Nếu `position_bias_delta` cao → judge đang bị ảnh hưởng bởi vị trí, không phải nội dung.

### Async + Semaphore
Không có semaphore → 60 case gọi API cùng lúc → bị rate limit. Với `asyncio.Semaphore(10)`, tối đa 10 requests chạy song song — đủ để hoàn thành 60 cases trong < 30 giây thay vì tuần tự mất 5+ phút.

### Release Gate
Gate có 4 điều kiện AND — tất cả phải thỏa:
1. `delta_avg_score >= +0.2` — V2 phải tốt hơn đáng kể
2. `delta hit_rate@3 >= -0.02` — retrieval không được tụt
3. `agreement_rate >= 0.70` — judge phải đáng tin
4. `p95_latency <= 1.2 × V1` — không chậm hơn nhiều

Trade-off: ngưỡng quá cao → không bao giờ release; quá thấp → ship regression. Ngưỡng `+0.2` score cho phép cải tiến nhỏ nhưng có ý nghĩa thống kê.

---

## 3. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: Agent stub không trả `retrieved_ids`
- **Symptom:** Retrieval eval trả về `hit_rate = 0` cho mọi case, vì agent không trả `retrieved_ids`.  
- **Root cause:** `MainAgent.query()` gốc chỉ trả `answer` và `contexts`, không có `retrieved_ids`.  
- **Fix:** Thêm keyword-based retrieval vào agent, map về đúng `doc_id` của golden dataset.

### Vấn đề 2: V1 và V2 cùng điểm → gate luôn BLOCK
- **Root cause:** Agent cũ không có V1/V2 mode, cả hai chạy như nhau.  
- **Fix:** V1 chỉ dùng context đầu tiên; V2 dùng tất cả context chunks → V2 cho câu trả lời đầy đủ hơn → judge cho điểm cao hơn.

### Vấn đề 3: Mock mode cần reproducible
- Khi không có API key, dùng `hashlib.md5(salt)` để tạo điểm deterministic — cùng case luôn cho cùng điểm, test suite có thể verify được.

---

## 4. Trade-off quan trọng

| Trade-off | Lựa chọn | Lý do |
|---|---|---|
| Cost vs Reliability | Dùng `gpt-4o-mini` x2 + `gpt-4o` tiebreaker | Tiết kiệm chi phí bình thường, chỉ dùng model đắt khi conflict |
| Throughput vs Rate Limit | Semaphore = 10 | Đủ nhanh (<2 phút/60 cases), không bị 429 |
| Strictness của gate | delta ≥ +0.2 | Không quá dễ (>0.0) cũng không quá khắt khe (>0.5) |
| Position bias check | Mỗi case thêm 1 call | Tăng cost 33% nhưng phát hiện bias là điều kiện điểm cá nhân |

---

## 5. Điều tôi sẽ làm khác nếu có thêm thời gian

1. **Judge caching**: Cache kết quả theo hash(question + answer) để tránh gọi API lặp khi rerun với cùng dataset — có thể giảm 30%+ cost eval.
2. **Judge cascade**: Chỉ gọi model đắt (`gpt-4o`) cho các case có uncertainty cao (score gần ngưỡng pass/fail), dùng `gpt-4o-mini` cho phần còn lại.
3. **Retry với exponential jitter**: Thêm random jitter vào sleep time để tránh thundering herd khi nhiều request retry cùng lúc.
4. **Kết nối agent với vector DB thật**: `_retrieve()` hiện tại là keyword matching — cần thay bằng FAISS/Chroma lookup để hit_rate có ý nghĩa thực sự.
