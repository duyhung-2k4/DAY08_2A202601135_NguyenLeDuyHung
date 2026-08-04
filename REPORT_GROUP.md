# REPORT_GROUP - Báo Cáo Phân Công Nhóm

## 1. Thông tin chung

Nhóm thực hiện bài lab theo **Phương án B: Nhóm 5 thành viên - Chuyên sâu Retrieval**. Hướng phân công này tách riêng các phần chính của hệ thống RAG để các thành viên có thể làm song song, sau đó tích hợp lại thành một pipeline hoàn chỉnh.

Chủ đề của dự án là xây dựng hệ thống **University Services RAG Chatbot**, hỗ trợ trả lời câu hỏi về chính sách và dịch vụ đại học như học phí, học bổng, ký túc xá, thư viện, đăng ký học phần và các thông báo liên quan.

## 2. Kiến trúc tổng quan

Pipeline của nhóm được thiết kế theo luồng sau:

```text
Data Collection
    -> Markdown Standardization
    -> Chunking & ChromaDB Indexing
    -> Semantic Search + Lexical Search
    -> RRF Reranking
    -> PageIndex Fallback
    -> Generation with Citation
    -> Streamlit Chatbot / Evaluation
```

Trong đó, phần Retrieval được chia sâu thành Dense Search, Sparse Search, Reranking và Fallback để tăng chất lượng truy xuất tài liệu trước khi sinh câu trả lời.

## 3. Phân công công việc

| Thành viên | MSSV | Vai trò | Công việc chính | File / Task phụ trách | Trạng thái |
|---|---|---|---|---|---|
| Nguyễn Lê Duy Hưng | 2A202601135 | Role 1 - Team Leader & RAG Architect | Quản lý tiến độ, thiết kế kiến trúc RAG tổng thể, điều phối tích hợp code và kiểm tra pipeline chính | `src/task9_retrieval_pipeline.py`, `src/supervisor.py`, tài liệu nhóm | Hoàn thành |
| Trương Công Đạt | 2A202601449 | Role 2 - Data & Dense Search Dev | Thu thập dữ liệu, chuẩn hóa dữ liệu, xây dựng ChromaDB và semantic search | Task 1, Task 2, Task 3, Task 4, Task 5 | Hoàn thành |
| Hồ Phạm Đức Linh | 2A202601533 | Role 3 - Sparse Search & Advanced Reranking Dev | Xây dựng lexical search bằng BM25/TF-IDF, gộp kết quả bằng RRF và hỗ trợ fallback vectorless | Task 6, Task 7, Task 8 | Hoàn thành |
| Nguyễn Thị Phương | 2A202601315 | Role 4 - Frontend & Chatbot Developer | Xây dựng giao diện chatbot bằng Streamlit, kết nối pipeline truy xuất với phần sinh câu trả lời có citation | `app.py`, Task 10 | Hoàn thành |
| Nguyễn Văn Minh | 2A202601972 | Role 5 - Evaluation & QA Engineer | Tạo bộ câu hỏi đánh giá, chạy đánh giá RAGAS, so sánh các cấu hình retrieval và viết báo cáo kết quả | `group_project/evaluation/golden_dataset.json`, `eval_pipeline.py`, `results.md` | Hoàn thành |

## 4. Mô tả chi tiết theo từng vai trò

### Role 1 - Team Leader & RAG Architect

Nguyễn Lê Duy Hưng phụ trách điều phối chung, chia task cho các thành viên và đảm bảo các phần code có thể tích hợp với nhau. Vai trò này tập trung vào thiết kế pipeline RAG tổng thể, đặc biệt là Task 9 - kết nối semantic search, lexical search, RRF reranking và PageIndex fallback.

Ngoài ra, Role 1 kiểm tra logic fallback để đảm bảo hệ thống không dùng điểm RRF làm ngưỡng đánh giá độ liên quan. Thay vào đó, pipeline sử dụng điểm semantic gốc từ `semantic_search()` để quyết định khi nào cần chuyển sang PageIndex fallback.

### Role 2 - Data & Dense Search Dev

Đạt phụ trách phần dữ liệu và dense retrieval. Công việc bao gồm thu thập tài liệu chính sách/dịch vụ đại học, crawl bài viết hoặc thông báo, chuyển đổi dữ liệu sang Markdown, chia nhỏ văn bản thành chunk và xây dựng vector store bằng ChromaDB.

Sau khi dữ liệu được index, Đạt triển khai semantic search để truy xuất các đoạn văn bản có ý nghĩa gần với câu hỏi của người dùng. Đây là nguồn retrieval chính cho pipeline RAG.

### Role 3 - Sparse Search & Advanced Reranking Dev

Linh phụ trách phần sparse retrieval và reranking nâng cao. Công việc chính là xây dựng lexical search bằng BM25/TF-IDF để tìm các đoạn văn bản khớp từ khóa chính xác, đặc biệt hữu ích với tên riêng, mã tài liệu, số hiệu hoặc thuật ngữ cụ thể.

Linh cũng triển khai RRF reranking để gộp kết quả từ semantic search và lexical search. Ngoài ra, Role 3 hỗ trợ phần PageIndex fallback để hệ thống vẫn có phương án truy xuất khi hybrid retrieval chưa tìm được kết quả đủ tốt.

### Role 4 - Frontend & Chatbot Developer

Phương phụ trách giao diện người dùng bằng Streamlit và tích hợp chatbot với pipeline phía sau. Giao diện cho phép người dùng nhập câu hỏi, điều chỉnh số lượng chunk truy xuất và xem danh sách nguồn tài liệu được sử dụng.

Phương cũng hỗ trợ phần generation có citation, đảm bảo câu trả lời cuối cùng không chỉ có nội dung tổng hợp mà còn hiển thị nguồn tham khảo rõ ràng.

### Role 5 - Evaluation & QA Engineer

Minh phụ trách kiểm thử và đánh giá chất lượng hệ thống RAG. Công việc bao gồm xây dựng bộ câu hỏi đánh giá `golden_dataset.json`, chạy evaluation bằng RAGAS và ghi nhận các chỉ số như faithfulness, answer relevance, context recall và context precision.

Minh cũng thực hiện so sánh A/B giữa các cấu hình retrieval, ví dụ hybrid search so với dense-only hoặc có reranking so với không reranking, từ đó đưa ra nhận xét về điểm mạnh, điểm yếu và hướng cải thiện của hệ thống.

## 5. Cách nhóm phối hợp

Nhóm chia công việc theo module để giảm conflict khi code. Mỗi thành viên tập trung vào file hoặc task được giao, sau đó đẩy code lên repository chung. Role 1 phụ trách kiểm tra phần tích hợp cuối cùng, đặc biệt là các file dùng chung như `app.py`, `src/task9_retrieval_pipeline.py`, `requirements.txt` và thư mục `group_project/evaluation/`.

Trong quá trình tích hợp, nhóm ưu tiên kiểm tra các tiêu chí sau:

- Các function trả về đúng format `list[dict]` với các trường `content`, `score`, `metadata`.
- Retrieval pipeline không bị crash khi một module chưa sẵn sàng hoặc chưa có API key.
- Kết quả chatbot có citation và có thể hiển thị source documents.
- Evaluation pipeline có dữ liệu golden dataset và báo cáo kết quả rõ ràng.

## 6. Kết quả đạt được

Nhóm đã hoàn thành các phần chính của pipeline RAG theo phương án B. Hệ thống có khả năng thu thập và chuẩn hóa dữ liệu, index tài liệu vào vector store, truy xuất bằng cả semantic search và lexical search, gộp kết quả bằng RRF, hỗ trợ fallback bằng PageIndex và tích hợp vào chatbot Streamlit.

Bên cạnh đó, nhóm cũng chuẩn bị phần đánh giá RAGAS để đo chất lượng câu trả lời và chất lượng retrieval. Cách phân công theo vai trò giúp các thành viên làm song song, hạn chế trùng lặp công việc và thuận lợi hơn khi tích hợp sản phẩm cuối cùng.

