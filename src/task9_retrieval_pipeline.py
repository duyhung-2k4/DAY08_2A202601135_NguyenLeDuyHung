"""
Task 9 â€” Retrieval Pipeline HoÃ n Chá»‰nh.

Káº¿t há»£p semantic search + lexical search + reranking + PageIndex fallback
thÃ nh má»™t pipeline thá»‘ng nháº¥t.

Logic:
    1. Cháº¡y semantic_search + lexical_search song song
    2. Merge káº¿t quáº£ (RRF hoáº·c weighted fusion)
    3. Rerank
    4. Náº¿u top result score < threshold â†’ fallback sang PageIndex
    5. Return top_k results

âš ï¸ BáºªY THÆ¯á»œNG Gáº¶P â€” Ä‘á»c ká»¹ trÆ°á»›c khi code:
    Náº¿u báº¡n dÃ¹ng Ä‘iá»ƒm RRF Ä‘Ã£ fuse (Task 7) Ä‘á»ƒ so vá»›i score_threshold, báº¡n sáº½ gáº·p bug
    tháº­t: RRF max score luÃ´n â‰ˆ 1/(k+1) â‰ˆ 0.0164 (k=60) Báº¤T Ká»‚ ná»™i dung cÃ³ liÃªn quan
    hay khÃ´ng. Náº¿u Ä‘áº·t threshold tháº¥p (nhÆ° 0.005) Ä‘á»ƒ "há»£p" vá»›i thang Ä‘iá»ƒm RRF, thá»±c
    cháº¥t KHÃ”NG cÃ¢u há»i nÃ o Ä‘á»§ tháº¥p Ä‘á»ƒ trigger fallback ná»¯a â€” ká»ƒ cáº£ query hoÃ n toÃ n vÃ´
    nghÄ©a váº«n tráº£ vá» káº¿t quáº£ "hybrid" (rÃ¡c) thay vÃ¬ fallback Ä‘Ãºng nhÆ° thiáº¿t káº¿.

    CÃ¡ch sá»­a Ä‘Ãºng: giá»¯ Ä‘iá»ƒm cosine similarity Gá»C cá»§a semantic_search (trÆ°á»›c khi qua
    RRF) lÃ m cÄƒn cá»© quyáº¿t Ä‘á»‹nh fallback, tÃ¡ch biá»‡t khá»i Ä‘iá»ƒm RRF dÃ¹ng Ä‘á»ƒ sáº¯p xáº¿p káº¿t
    quáº£ cuá»‘i cÃ¹ng. Calibrate threshold báº±ng cÃ¡ch tá»± Ä‘o: cháº¡y vÃ i cÃ¢u há»i cháº¯c cháº¯n
    liÃªn quan vÃ  vÃ i cÃ¢u cháº¯c cháº¯n láº¡c Ä‘á»/rÃ¡c qua semantic_search, xem khoáº£ng cÃ¡ch
    Ä‘iá»ƒm sá»‘ giá»¯a hai nhÃ³m rá»“i chá»n ngÆ°á»¡ng náº±m giá»¯a.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold nÃ y báº±ng cÃ¡ch tá»± Ä‘o Ä‘iá»ƒm cosine cá»§a semantic_search
# cho cÃ¢u há»i liÃªn quan vs cÃ¢u há»i láº¡c Ä‘á» (xem ghi chÃº á»Ÿ trÃªn) â€” Äá»ªNG copy nguyÃªn
# giÃ¡ trá»‹ máº«u, má»—i corpus/embedding model sáº½ cho khoáº£ng Ä‘iá»ƒm khÃ¡c nhau.
SCORE_THRESHOLD = 0.3   # Náº¿u best score (cosine gá»‘c) < threshold â†’ fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoÃ n chá»‰nh vá»›i fallback logic.

    Pipeline:
        Query
          â”œâ†’ Semantic Search â†’ dense_results (giá»¯ Ä‘iá»ƒm cosine gá»‘c)
          â”œâ†’ Lexical Search  â†’ sparse_results
          â”‚
          â”œâ†’ Merge (RRF) â†’ merged_results
          â”œâ†’ Rerank â†’ reranked_results
          â”‚
          â””â†’ If dense_results[0]["score"] < threshold:
                â””â†’ PageIndex Vectorless â†’ fallback_results

    Args:
        query: CÃ¢u truy váº¥n
        top_k: Sá»‘ lÆ°á»£ng káº¿t quáº£ cuá»‘i cÃ¹ng
        score_threshold: NgÆ°á»¡ng Ä‘iá»ƒm cosine gá»‘c tá»‘i thiá»ƒu (KHÃ”NG pháº£i Ä‘iá»ƒm RRF)
        use_reranking: CÃ³ Ã¡p dá»¥ng reranking hay khÃ´ng

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoáº·c 'pageindex'
        }
    """
    candidate_k = max(top_k * 2, top_k)

    try:
        dense_results = semantic_search(query, top_k=candidate_k)
    except Exception as exc:
        print(f"âš  semantic_search lá»—i trong retrieve ({type(exc).__name__}: {exc})")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=candidate_k)
    except Exception as exc:
        print(f"âš  lexical_search lá»—i trong retrieve ({type(exc).__name__}: {exc})")
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
            print(f"âš  rerank lá»—i trong retrieve ({type(exc).__name__}: {exc})")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item["source"] = "hybrid"

    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(
            f"  âš  Semantic best score ({best_score:.3f}) < threshold ({score_threshold})"
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as exc:
            print(f"âš  pageindex_search khÃ´ng sáºµn sÃ ng ({type(exc).__name__}: {exc})")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query khÃ´ng cÃ³ káº¿t quáº£ â†’ test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
