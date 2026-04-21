# Individual Reflection — VoThanhDanh (Role 1: Data / Retrieval / Failure Analysis)

**Họ và tên:** Võ Thành Danh  
**MSSV:** 2A202600503  
**Vai trò:** Role 1 (Data / Retrieval / Failure Analysis)

## 1. Công việc tôi đã làm

### Module chính (ownership của Role 1)

| File | Mô tả công việc |
|---|---|
| `data/synthetic_gen.py` | Thiết kế và sinh Golden Dataset 60 cases theo phân phối chuẩn (normal, adversarial, edge, conflicting, multi-turn), chuẩn hóa schema (`case_id`, `expected_retrieval_ids`, `case_type`, `difficulty`, `tags`) |
| `engine/retrieval_eval.py` | Triển khai retrieval metrics đầy đủ: `HitRate@1/@3/@5`, `MRR`, `evaluate_case`, `evaluate_batch`, `top_k_sensitivity`, correlation retrieval↔answer (tùy chọn) |
| `analysis/failure_analysis.md` | Cập nhật báo cáo nhóm phần Role 1 bằng số liệu run thật: top-k sensitivity, hard-case retrieval fail-rate, failure clustering, 5 Whys cho case fail thật |
| `tests/test_synthetic_gen.py` | Unit test cho phân phối dataset, schema validity, hard-case pack coverage |
| `tests/test_retrieval_eval.py` | Unit test cho công thức HitRate/MRR và batch aggregation |

### Kết quả thực nghiệm chính (run thật có API key)

Nguồn: `reports/summary.json`, `reports/benchmark_results.json`

- Top-k sensitivity:
  - `HitRate@1 = 0.7333`
  - `HitRate@3 = 0.8667`
  - `HitRate@5 = 0.8667`
  - `MRR = 0.7917`
- Hard-case retrieval fail-rate (định nghĩa fail = `hit_rate@3 = 0`):
  - `8 / 30 = 26.67%` (vượt ngưỡng 20% cho stress-test retrieval)
- Hard-case status fail-rate (tham khảo):
  - `5 / 30 = 16.67%`

---

## 2. Các khái niệm kỹ thuật tôi đã hiểu sâu

### 2.1. MRR (Mean Reciprocal Rank)
- **Định nghĩa:** MRR đo chất lượng thứ hạng của retriever thông qua vị trí tài liệu đúng đầu tiên trong danh sách trả về.
- **Công thức per-query:** nếu tài liệu đúng đầu tiên ở vị trí `r` (đánh số từ 1), điểm là `1/r`; nếu không có tài liệu đúng, điểm là `0`.
- **Công thức toàn tập:**  

```text
MRR = (1/N) * Σ(1/r_i)
```

- **Ý nghĩa thực tế trong RAG:** MRR nhấn mạnh chất lượng top đầu. Tài liệu đúng xuất hiện sớm (top-1/top-2) giúp giảm xác suất model dùng context sai, từ đó giảm nguy cơ hallucination và câu trả lời lạc đề.
- **So sánh với HitRate@K:** HitRate@K cho biết “có tìm thấy hay không”, còn MRR phản ánh “tìm thấy sớm tới mức nào”. Vì vậy MRR hữu ích khi cần tối ưu ranking quality thay vì chỉ tối ưu recall thô.

### 2.2. Cohen's Kappa
- **Định nghĩa:** Cohen’s Kappa đo mức đồng thuận giữa hai bộ chấm điểm sau khi đã loại trừ phần đồng thuận xảy ra do ngẫu nhiên.
- **Công thức:**  

```text
kappa = (p_o - p_e) / (1 - p_e)
```

  - `p_o`: tỷ lệ đồng thuận quan sát được  
  - `p_e`: tỷ lệ đồng thuận kỳ vọng nếu hai bên chấm ngẫu nhiên theo phân phối nhãn
- **Diễn giải chuẩn:**  
  - `kappa ~ 1`: đồng thuận mạnh, hệ thống chấm ổn định  
  - `kappa ~ 0`: đồng thuận gần mức ngẫu nhiên  
  - `kappa < 0`: bất đồng có hệ thống
- **Giá trị kỹ thuật:** trong bài toán Multi-Judge, Kappa giúp tránh “agreement ảo” khi một nhãn chiếm đa số. Đây là chỉ số cần thiết để đánh giá độ tin cậy thực của hệ thống đánh giá.

### 2.3. Position Bias
- **Định nghĩa:** Position Bias là hiện tượng judge đánh giá tốt hơn cho phương án xuất hiện ở vị trí đầu (hoặc vị trí cố định), ngay cả khi chất lượng hai phương án tương đương.
- **Rủi ro:** nếu không kiểm soát bias này, kết quả benchmark có thể phản ánh thứ tự hiển thị thay vì phản ánh chất lượng thực.
- **Cách kiểm tra chuẩn:** thực hiện pairwise evaluation hai lượt:
  - Lượt 1: chấm theo thứ tự A/B
  - Lượt 2: hoán đổi B/A
  - So sánh độ lệch điểm và tỷ lệ đổi winner giữa hai lượt.
- **Hướng xử lý:** randomize thứ tự đầu vào, chấm lặp nhiều lần và tổng hợp bằng thống kê (mean/median + variance) để giảm sai lệch do vị trí.

### 2.4. Trade-off Chi phí và Chất lượng
- **Quan sát kỹ thuật:** tăng `top_k` thường cải thiện xác suất chứa tài liệu đúng trong context, nhưng không miễn phí.
- **Chi phí phát sinh khi tăng `top_k`:**
  - tăng số chunk đưa vào prompt,
  - tăng token input và chi phí inference,
  - tăng latency do context dài hơn.
- **Rủi ro chất lượng:** nếu không có cơ chế reranking/lọc nhiễu, mở rộng `top_k` có thể kéo theo nhiều đoạn ít liên quan, làm model suy luận kém chính xác hơn.
- **Nguyên tắc tối ưu:** không tối ưu một chiều. Cần theo dõi đồng thời nhóm chỉ số chất lượng (`HitRate@K`, `MRR`, độ chính xác đầu ra) và nhóm chỉ số vận hành (token usage, latency, cost/request) trước khi chốt cấu hình retrieval.

---

## 3. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: Rủi ro lệch chuẩn dữ liệu SDG
- **Symptom:** Dữ liệu sinh tự động dễ thiếu field hoặc lệch phân phối.
- **Root cause:** Thiếu bước kiểm định bắt buộc trước khi ghi output.
- **Fix:** Viết `validate_dataset_schema` để chặn:
  - thiếu field bắt buộc,
  - sai kiểu dữ liệu,
  - lệch distribution so với kế hoạch.

### Vấn đề 2: Encoding lỗi trên Windows console
- **Symptom:** Script crash khi in một số ký tự Unicode.
- **Root cause:** Code page terminal không ổn định giữa môi trường.
- **Fix:** Chuẩn hóa log output sang ASCII cho các script chạy CLI.

### Vấn đề 3: Tránh claim bonus bằng số liệu không kiểm chứng
- **Symptom:** Có nguy cơ nhầm giữa số mô phỏng và số benchmark thật.
- **Root cause:** Thiếu nguyên tắc evidence-first trong báo cáo ban đầu.
- **Fix:** Chỉ dùng số trực tiếp từ `reports/summary.json` và `reports/benchmark_results.json`, loại bỏ toàn bộ phần không traceable.

### Vấn đề 4: Đồng bộ bonus Role 1 sau khi pull Role 2
- **Symptom:** Trước khi hợp nhất Role 2, chỉ có framework Role 1, chưa có số liệu end-to-end.
- **Root cause:** Bonus Role 1 phụ thuộc kết quả pipeline chạy thật (`main.py`) và output reports.
- **Fix:** Pull `main`, chạy lại `synthetic_gen -> main -> check_lab`, trích số từ report và cập nhật lại failure report + reflection.
- **Kết quả:** Bonus Role 1 được chốt bằng số liệu thực nghiệm, không dùng số mô phỏng.

### 3.2. Bài học rút ra
- Trong hệ thống đánh giá AI, sai lệch nhỏ ở dữ liệu hoặc cách diễn giải metric có thể dẫn đến kết luận sai toàn pipeline.
- Quy trình đúng cần ưu tiên: **schema-first, metric-first, evidence-first**.
- Mọi claim về chất lượng/bonus chỉ nên công bố khi có run thật và có artifact truy xuất được.

---

## 4. Trade-off quan trọng

| Trade-off | Lựa chọn | Lý do |
|---|---|---|
| Hard-case fail định nghĩa theo status hay retrieval | Dùng retrieval fail (`hit_rate@3=0`) cho bonus Role 1 | Bonus Role 1 tập trung vào stress-test retrieval stage, tách khỏi judge noise |
| Mở rộng top_k hay tối ưu ranking | Ưu tiên tối ưu ranking top đầu | `Hit@5` không tăng so với `Hit@3`, mở rộng thêm context không mang lại lợi ích rõ |
| Bao phủ test rộng vs tốc độ triển khai | Viết unit test tập trung cho data + metrics | Đảm bảo thay đổi ở SDG/retrieval có regression guard trước khi tích hợp pipeline |

---

## 5. Điều tôi sẽ làm khác nếu có thêm thời gian

1. **Semantic retrieval cho hard-cases**: thay keyword-heavy retrieval bằng embedding search để giảm miss ở nhóm edge/conflicting/multi-turn.
2. **Intent-aware query rewriting**: bóc tách intent trước retrieval (đặc biệt cho ambiguous queries).
3. **Per-cluster dashboard**: theo dõi riêng hit/mrr/fail-rate theo từng cluster để tối ưu có trọng tâm hơn.
4. **Automatic role1 bonus checker**: script tự tính top-k sensitivity + hard-case retrieval fail-rate sau mỗi run và append vào report.

---

## 6. Cam kết trung thực dữ liệu
- Tôi cam kết:
  - Không dùng số liệu mô phỏng để khai báo thành tích benchmark.
  - Các số liệu trong reflection này đều lấy từ run thật của pipeline hiện tại.
  - Nếu pipeline/weight/model thay đổi, tôi sẽ chạy lại và cập nhật số liệu tương ứng trước khi nộp bản cuối.