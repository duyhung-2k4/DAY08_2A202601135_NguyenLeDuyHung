"""
Task 9 - Complete Retrieval Pipeline.

Combine semantic search, lexical search, reranking, and PageIndex fallback
into one retrieval function.

Pipeline:
    1. Run semantic_search and lexical_search.
    2. Merge results with RRF.
    3. Rerank the merged results.
    4. If the original semantic score is below threshold, try PageIndex fallback.
    5. Return top_k results.

Important:
    Do not compare score_threshold with the RRF score. RRF scores are rank-based,
    not true relevance scores. Use the original semantic_search cosine score for
    fallback decisions.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
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
    Complete retrieval pipeline with fallback logic.

    Args:
        query: User query.
        top_k: Number of final results.
        score_threshold: Minimum original semantic score for hybrid retrieval.
        use_reranking: Whether to apply reranking after RRF merge.

    Returns:
        List of dicts with:
            content: result text
            score: retrieval score
            metadata: source metadata
            source: "hybrid" or "pageindex"
    """
    candidate_k = max(top_k * 2, top_k)

    try:
        dense_results = semantic_search(query, top_k=candidate_k)
    except Exception as exc:
        print(f"semantic_search failed in retrieve ({type(exc).__name__}: {exc})")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=candidate_k)
    except Exception as exc:
        print(f"lexical_search failed in retrieve ({type(exc).__name__}: {exc})")
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
            print(f"rerank failed in retrieve ({type(exc).__name__}: {exc})")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item["source"] = "hybrid"

    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(f"Semantic best score ({best_score:.3f}) < threshold ({score_threshold})")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as exc:
            print(f"pageindex_search is not ready ({type(exc).__name__}: {exc})")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
