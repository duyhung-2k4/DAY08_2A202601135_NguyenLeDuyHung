# RAG Evaluation Results

## Framework & Evaluator Stack

> **RAGAS v0.1.21** tích hợp với **GPT-4o (temperature=0.0)** bọc qua `LangchainLLMWrapper` làm LLM Judge, cùng Embedding Evaluator (`sentence-transformers/all-MiniLM-L6-v2` / `OpenAIEmbeddings`).
> **Benchmark Dataset**: [golden_dataset_vnu_15_ragas_0_1_21.json](file:///c:/Users/USER/Desktop/Vin/DAY08_2A202601135_NguyenLeDuyHung/group_project/evaluation/golden_dataset_vnu_15_ragas_0_1_21.json) (15 câu hỏi chuẩn hóa về quy chế & dịch vụ ĐHQGHN - VNU).

---

## Execution Summary

| Config | Total Samples | Successful Samples | Pipeline Errors | Success Rate |
|--------|--------------|-------------------|-----------------|--------------|
| **Config A (hybrid + rerank)** | 15 | 15 | 0 | 100.0% |
| **Config B (dense-only)** | 15 | 15 | 0 | 100.0% |

*Tất cả 15/15 test cases đã hoàn tất đánh giá không phát sinh lỗi thực thi pipeline.*

---

## Overall RAGAS Quality Scores & A/B Comparison

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A - B) | Đánh Giá Hiệu Năng |
|--------|---------------------------|----------------------|-----------|--------------------|
| **Faithfulness** | **0.8111** | 0.6405 | **+0.1706** (+17.06%) | **Config A vượt trội**: Giảm thiểu trung thực việc hallucination nhờ Reranker lọc nhiễu |
| **Answer Relevancy** | **0.6857** | 0.6249 | **+0.0608** (+6.08%) | Câu trả lời bám sát đúng trọng tâm yêu cầu |
| **Context Recall** | **0.8333** | 0.7333 | **+0.1000** (+10.00%) | Sparse BM25 hỗ trợ tìm đúng tài liệu chứa từ viết tắt/từ khóa |
| **Context Precision** | 0.6981 | **0.7133** | -0.0152 (-1.52%) | BM25 kéo theo một số chunk phụ chứa từ khóa lặp lại ở top-k |

---

## Direct Retrieval Metrics (Chỉ Số Đánh Giá Bộ Truy Vấn Trực Tiếp)

| Retrieval Metric | Config A (hybrid + rerank) | Config B (dense-only) | Ý Nghĩa Kỹ Thuật |
|------------------|---------------------------|----------------------|------------------|
| **Hit Rate@k** | **86.67%** (13/15) | **86.67%** (13/15) | Tỷ lệ lấy được đúng file tài liệu nguồn chứa đáp án ở top-k |
| **Recall@k** | **30.00%** | **30.00%** | Tỷ lệ trích xuất trọn vẹn từng chuỗi văn bản ground truth |
| **MRR (Mean Reciprocal Rank)** | 0.2000 | **0.2333** | Thứ hạng vị trí đầu tiên của chunk chính xác nhất |

---

## Phân Tích Điểm Theo Nhóm Câu Hỏi (Category Breakdown)

| Nhóm Câu Hỏi (Category) | Số lượng | Config A (Overall) | Config B (Overall) | Nhận Xét Đột Phá |
|-------------------------|----------|-------------------|-------------------|------------------|
| **Happy Case** | 5 | **0.8875** | 0.8384 | Truy vấn tiêu chuẩn đạt độ chính xác gần như tuyệt đối (~0.99 ở Q1, Q4) |
| **Paraphrase / Synonym** | 3 | **0.8016** | 0.6117 | **Config A tăng +18.99%**: Nhờ BM25 khớp được từ viết tắt (SV, CTĐT, KTX, GPA) |
| **Edge Case** | 3 | **0.8969** | 0.8148 | Cải thiện xử lý các câu hỏi chứa logic điều kiện phức tạp (tín chỉ tối thiểu, học bằng 2) |
| **Out of Domain** | 2 | **0.5625** | 0.5000 | Kích hoạt đúng Guardrail từ chối ("Tôi không thể xác minh thông tin này"), Faithfulness đạt 1.0 |
| **Multilingual (Cross-lingual)** | 2 | 0.3490 | **0.3493** | Thách thức lớn nhất do bất đồng ngôn ngữ Anh - Việt giữa query và corpus quy chế |

---

## Worst Performers (Phân Tích 3 Câu Hỏi Thấp Điểm Nhất)

| # | Question ID & Nội Dung Câu Hỏi | Faithfulness | Relevance | Recall | Precision | Stage Thất Bại | Nguyên Nhân Gốc Rễ (Root Cause) |
|---|--------------------------------|--------------|-----------|--------|-----------|----------------|---------------------------------|
| **1** | **[vnu_eval_014]** *What is the minimum weight of the final course assessment in the overall course grade at VNU?* | 0.00 | 0.00 | 0.00 | 0.00 | **Retrieval (Cross-lingual)** | BM25 không thể khớp từ tiếng Anh với văn bản quy chế tiếng Việt (`điểm thi kết thúc học phần tối thiểu 50%`). Dense Embedding `bge-m3` bị trôi rank do query tiếng Anh quá ngắn. |
| **2** | **[vnu_eval_012]** *Học phí ngành Công nghệ thông tin UET năm học 2026-2027 là bao nhiêu?* | 1.00 | 0.00 | 1.00 | 0.00 | **Evaluation Metric (Guardrail)** | Dữ liệu chỉ có năm 2025-2026. Pipeline từ chối đúng theo thiết kế guardrail (`I cannot verify...`), nhưng chỉ số RAGAS Relevancy phạt 0.0 do câu từ chối không lặp lại từ khóa query. |
| **3** | **[vnu_eval_003]** *Sinh viên đại học hệ chính quy cần đáp ứng tiêu chí chung nào để được xét học bổng khuyến khích học tập?* | 1.00 | 0.83 | 0.00 | 0.50 | **Retrieval (Context Recall)** | Tên tiêu đề trong quy chế dùng cụm `Đối tượng và tiêu chuẩn xét`, trong khi query dùng `tiêu chí chung`, khiến retrieval ưu tiên lấy chunk điều khoản chung thay vì chunk chứa 4 điều kiện cụ thể. |

---

## Recommendations & Đề Xuất Cải Tiến Pipeline

### Cải tiến 1: Tích hợp Multi-Query Expansion & Translation cho Cross-Lingual
* **Hành động**: Thêm bước tự động dịch query tiếng Anh sang tiếng Việt và sinh 2-3 câu hỏi biến thể (Query Rewriting) trước khi đưa vào Hybrid Retrieval.
* **Tác động dự kiến**: Khắc phục triệt để lỗi ở câu `vnu_eval_014` và tăng `Context Recall` từ 0.3490 lên **>0.80** cho các truy vấn tiếng Anh.

### Cải tiến 2: Tối ưu Chunking Strategy & Header-aware Indexing
* **Hành động**: Bổ sung metadata về tiêu đề Điều/Khoản (`Điều 5. Học phần`, `Điều 12. Học bổng KKHT`) vào từng chunk thay vì chỉ cắt theo độ dài `CHUNK_SIZE=800`.
* **Tác động dự kiến**: Khắc phục hiện tượng trôi chunk ở các câu hỏi như `vnu_eval_003`, tăng `Context Recall` và `Context Precision`.

### Cải tiến 3: Tinh chỉnh Threshold & Guardrail Handling
* **Hành động**: Tự động chuyển ngữ cảnh từ chối thành dạng chuẩn "Thông tin về [Query Subject] không được tìm thấy trong dữ liệu" để vừa duy trì tính trung thực vừa tối ưu điểm RAGAS Relevancy khi gặp Out-of-Domain queries.
* **Tác động dự kiến**: Tăng điểm tổng thể `Answer Relevancy` cho các câu hỏi thuộc nhóm Out-of-Domain.
