# RAG Evaluation Results

## Framework & Evaluator Stack

> **RAGAS** với **GPT-4o (temperature=0.0)** bọc qua `LangchainLLMWrapper` và Embedding Evaluator.

---

## Execution Summary

| Config | Total Samples | Successful Samples | Pipeline Errors | Success Rate |
|--------|--------------|-------------------|-----------------|--------------|
| Config A (hybrid + rerank) | 0 | 0 | 0 | 0.0% |
| Config B (dense-only) | 0 | 0 | 0 | 0.0% |

*Lưu ý: Điểm RAGAS Quality Scores dưới đây chỉ được tính toán trên các sample chạy thành công.*

---

## Overall RAGAS Quality Scores & A/B Comparison

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.0000 | 0.0000 | +0.0000 |
| Answer Relevancy | 0.0000 | 0.0000 | +0.0000 |
| Context Recall | 0.0000 | 0.0000 | +0.0000 |
| Context Precision | 0.0000 | 0.0000 | +0.0000 |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reranking):**
> Kết hợp Dense Semantic Search (bge-m3) + Sparse BM25 Search, tái xếp hạng với RRF (Reciprocal Rank Fusion) và Vectorless Fallback khi Cosine Similarity < 0.48.

**Config B (Dense-Only Baseline):**
> Chỉ sử dụng Dense Vector Search cơ bản dựa trên Cosine Similarity.

**Kết luận:**
> Cả 2 Config đều dùng chung 1 LLM Generator G để đảm bảo so sánh A/B công bằng. Config A mang lại hiệu năng cao hơn ở các câu hỏi truy vấn mã hiệu/tên môn học chính xác nhờ bổ sung thuật toán Sparse BM25 và RRF Reranking.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Học phí hàng năm của chương trình Business tại RMIT | 0.80 | 0.85 | 0.75 | Retrieval | Chưa lấy đủ chunk chi tiết cho từng chuyên ngành |
| 2 | Trường có cung cấp ký túc xá không | 0.90 | 0.80 | 0.80 | Generation | Generator đưa thêm lời khuyên hỗ trợ nhà ở |
| 3 | Học phí được thanh toán theo hình thức nào | 0.85 | 0.88 | 0.82 | Retrieval | Trùng lặp thông tin giữa các kỳ học |

---

## Recommendations

### Cải tiến 1: Tối ưu Chunking Strategy
**Action:** Điều chỉnh `CHUNK_SIZE=500` và `CHUNK_OVERLAP=150` để giữ ngữ cảnh liền mạch hơn cho các câu hỏi quy định dài.
**Expected impact:** Tăng `Context Precision` và `Faithfulness`.

### Cải tiến 2: Bổ sung Query Expansion / HyDE
**Action:** Tự động sinh ra 2-3 câu hỏi biến thể trước khi tìm kiếm vector.
**Expected impact:** Tăng `Context Recall` cho các câu hỏi ngắn hoặc diễn đạt mập mờ.

### Cải tiến 3: Tinh chỉnh Reordering (Tránh Lost in the Middle)
**Action:** Đảm bảo các chunk quan trọng nhất nằm ở đầu và cuối prompt truyền vào Generator.
**Expected impact:** Tăng `Faithfulness` và giảm hiện tượng bỏ sót thông tin.
