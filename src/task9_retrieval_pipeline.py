"""
Task 9 - Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search, lexical search, reranking và PageIndex fallback
thành một pipeline truy xuất thống nhất.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Gộp kết quả bằng RRF.
    3. Rerank lại danh sách đã gộp.
    4. Nếu điểm semantic gốc thấp hơn ngưỡng, thử fallback sang PageIndex.
    5. Trả về top_k kết quả cuối cùng.

Lưu ý quan trọng:
    Không dùng điểm RRF để so với score_threshold. Điểm RRF chỉ dựa trên thứ hạng,
    không phản ánh trực tiếp độ liên quan thật sự. Quyết định fallback phải dựa trên
    điểm cosine gốc từ semantic_search.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CẤU HÌNH
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Pipeline truy xuất hoàn chỉnh có logic fallback.

    Args:
        query: Câu truy vấn của người dùng.
        top_k: Số lượng kết quả cuối cùng cần trả về.
        score_threshold: Ngưỡng điểm semantic gốc tối thiểu để dùng hybrid retrieval.
        use_reranking: Có áp dụng reranking sau khi gộp bằng RRF hay không.

    Returns:
        List các dict có dạng:
            content: nội dung đoạn văn bản
            score: điểm truy xuất
            metadata: metadata nguồn tài liệu
            source: "hybrid" hoặc "pageindex"
    """
    candidate_k = max(top_k * 2, top_k)

    try:
        dense_results = semantic_search(query, top_k=candidate_k)
    except Exception as exc:
        print(f"semantic_search lỗi trong retrieve ({type(exc).__name__}: {exc})")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=candidate_k)
    except Exception as exc:
        print(f"lexical_search lỗi trong retrieve ({type(exc).__name__}: {exc})")
        sparse_results = []

    ranked_lists = [results for results in [dense_results, sparse_results] if results]
    merged: list[dict] = rerank_rrf(ranked_lists, top_k=candidate_k) if ranked_lists else []

    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except NotImplementedError:
            final_results = merged[:top_k]
        except Exception as exc:
            print(f"rerank lỗi trong retrieve ({type(exc).__name__}: {exc})")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item["source"] = "hybrid"

    # Dùng điểm semantic gốc để quyết định fallback, không dùng điểm RRF.
    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(f"Semantic best score ({best_score:.3f}) < threshold ({score_threshold})")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as exc:
            print(f"pageindex_search chưa sẵn sàng ({type(exc).__name__}: {exc})")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Học phí tại VNU là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Có những học bổng nào cho sinh viên quốc tế?",
        "xyzabc123nonsense",
    ]

    for q in test_queries:
        print(f"\nCâu hỏi: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
