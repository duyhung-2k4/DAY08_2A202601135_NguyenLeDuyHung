"""
Task 5 — Semantic Search Module (+ HyDE bonus).

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Bonus (README — "Implement phương pháp hỗ trợ Semantic Search"):
    `hyde_search()` bên dưới implement HyDE (Hypothetical Document Embeddings).

Chạy:
    python -m src.task5_semantic_search
"""

import os

from dotenv import load_dotenv

from .task4_chunking_indexing import get_collection, get_embedding_model

load_dotenv()

# Bật/tắt HyDE cho pipeline. Để False mặc định: HyDE cần gọi LLM nên chậm hơn và
# tốn quota (OpenRouter free chỉ 50 request/ngày/tài khoản). Task 9 / app.py có thể
# gọi trực tiếp hyde_search() khi muốn dùng.
USE_HYDE = False

# Tên model khi gọi thẳng OpenAI. Khi fallback qua OpenRouter, _generate_hypothetical_doc()
# tự thêm tiền tố "openai/" cần thiết cho routing của OpenRouter.
HYDE_MODEL = "gpt-4o-mini"


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (dense retrieval).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score [0, 1]
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending. Trả về [] nếu vector store chưa có dữ liệu.
    """
    try:
        collection = get_collection()

        # Chưa chạy Task 4 → collection rỗng. Phải thoát sớm: Chroma trả về mảng
        # rỗng và results["documents"][0] sẽ ném IndexError.
        if collection.count() == 0:
            return []

        # Bước 1: Embed query bằng đúng model đã dùng ở Task 4.
        # bge-m3 KHÔNG cần instruction prefix cho query (khác bge-large-en) —
        # encode thẳng chuỗi query.
        model = get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        # Bước 2: Query vector store bằng cosine similarity.
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Bước 3: Quy đổi distance → similarity và sắp xếp giảm dần.
        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]

    except Exception as exc:
        # Không để retrieval làm sập cả pipeline/chatbot: chưa index, chroma_db/ bị xoá,
        # model tải dở... đều trả về rỗng để Task 9 kích hoạt fallback.
        print(f"⚠ semantic_search lỗi ({type(exc).__name__}: {exc}) → trả về []")
        return []


# =============================================================================
# BONUS — HyDE (Hypothetical Document Embeddings)
# =============================================================================

def _generate_hypothetical_doc(query: str) -> str:
    """
    Nhờ LLM viết một đoạn trả lời GIẢ ĐỊNH cho query.

    Ý tưởng HyDE: câu hỏi ("Học phí bao nhiêu?") và văn bản quy định
    ("Điều 8. Mức thu học phí năm học 2025-2026 được xác định...") khác nhau rất xa
    về mặt từ vựng lẫn văn phong, nên embedding của chúng cũng xa nhau. Thay vì embed
    câu hỏi, ta embed một đoạn văn *trông giống tài liệu đích* → vector nằm gần vùng
    tài liệu thật hơn.

    Trả về "" nếu không gọi được LLM (không có API key, hết quota, lỗi mạng).
    """
    # Ưu tiên gọi thẳng OpenAI; fallback OpenRouter (free tier) nếu không có OPENAI_API_KEY.
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ""

    try:
        from openai import OpenAI

        using_openrouter = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1" if using_openrouter else None,
        )
        # OpenRouter cần tiền tố provider (vd "openai/gpt-4o-mini"); OpenAI trực tiếp thì không.
        model = f"openai/{HYDE_MODEL}" if using_openrouter else HYDE_MODEL

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn viết trích đoạn văn bản quy định của Đại học Quốc gia Hà Nội. "
                        "Với câu hỏi của sinh viên, hãy viết 2-3 câu theo đúng văn phong "
                        "văn bản hành chính/quy chế như thể đang trích từ tài liệu chính thức. "
                        "Chỉ trả về đoạn văn, không giải thích, không nói rằng bạn đang giả định."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,   # thấp: cần văn phong ổn định, không cần sáng tạo
            max_tokens=200,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"⚠ HyDE không sinh được đoạn giả định ({type(exc).__name__}) → dùng query gốc")
        return ""


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic search với HyDE.

    Tự động degrade về semantic_search() thường nếu không gọi được LLM —
    HyDE là bonus, không được phép làm hỏng pipeline chính.
    """
    hypothetical = _generate_hypothetical_doc(query)
    if not hypothetical:
        return semantic_search(query, top_k)

    # Ghép query gốc vào đoạn giả định thay vì thay thế hoàn toàn: HyDE thuần dễ
    # "trôi" khỏi các từ khoá chính xác (số hiệu quyết định 4618/QĐ-ĐHQGHN, tên riêng)
    # vì LLM có thể bịa số liệu. Giữ query gốc để neo lại các token đó.
    augmented = f"{hypothetical} {query}"

    try:
        collection = get_collection()
        if collection.count() == 0:
            return []

        model = get_embedding_model()
        query_vector = model.encode(augmented, normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            output.append({
                "content": doc,
                "score": round(max(0.0, 1.0 - dist), 4),
                "metadata": meta,
            })

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]

    except Exception as exc:
        print(f"⚠ hyde_search lỗi ({type(exc).__name__}) → fallback semantic_search")
        return semantic_search(query, top_k)


def search(query: str, top_k: int = 10) -> list[dict]:
    """Entry point cho Task 9 / app.py — tôn trọng cờ USE_HYDE."""
    return hyde_search(query, top_k) if USE_HYDE else semantic_search(query, top_k)


if __name__ == "__main__":
    # Smoke test: chạy cả tiếng Việt lẫn tiếng Anh để xác nhận bge-m3 cross-lingual
    # hoạt động — corpus là tiếng Việt nhưng test suite dùng query tiếng Anh.
    for q in ["học phí", "thủ tục đăng ký ký túc xá", "what is the tuition fee"]:
        print(f"\n{'=' * 60}\nQuery: {q}\n{'=' * 60}")
        for r in semantic_search(q, top_k=3):
            src = r["metadata"].get("source", "?")
            print(f"[{r['score']:.3f}] ({src}) {r['content'][:120].strip()}...")
