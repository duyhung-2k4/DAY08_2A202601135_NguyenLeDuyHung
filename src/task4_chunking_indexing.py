"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
`index_to_vectorstore()` bên dưới đã tự xử lý việc này (drop collection trước khi tạo lại).

Chạy:
    python -m src.task4_chunking_indexing
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter.
# Vì sao KHÔNG dùng MarkdownHeaderTextSplitter? Corpus của nhóm chủ yếu là PDF văn bản
# quy phạm (Quy chế đào tạo, Quy định học bổng ĐHQGHN) — MarkItDown trích ra text phẳng
# gần như KHÔNG có heading markdown (`#`, `##`), nên header splitter sẽ không cắt được gì
# và trả về nguyên một khối khổng lồ. Recursive luôn cắt được và đảm bảo trần kích thước.
CHUNKING_METHOD = "recursive"

# Vì sao 800? Một điều/khoản của văn bản quy phạm tiếng Việt (vd "Điều 12. Học bổng
# khuyến khích học tập") thường dài 400–800 ký tự. Chọn 800 để một chunk chứa trọn ý,
# không bị cắt giữa điều kiện xét duyệt. Nhỏ hơn (500) làm vỡ ngữ cảnh điều khoản;
# lớn hơn (1500) làm loãng embedding, cosine score tụt đều cho mọi query.
CHUNK_SIZE = 800

# Vì sao 100 (12.5%)? Đủ để một câu bị cắt ngang vẫn xuất hiện trọn vẹn ở chunk kế tiếp,
# nhưng chưa tới mức nhân đôi số chunk (overlap lớn → chi phí embedding và trùng lặp
# kết quả retrieval tăng).
CHUNK_OVERLAP = 100

# Embedding model: BAAI/bge-m3.
# Vì sao? (1) Multilingual thật sự — corpus là tiếng Việt nhưng câu hỏi test và app có
# thể là tiếng Anh ("tuition fee"); bge-m3 cross-lingual nên vẫn match được.
# (2) Vượt trội all-MiniLM-L6-v2 trên tiếng Việt (MiniLM huấn luyện chủ yếu tiếng Anh).
# (3) Chạy local, không tốn API quota như text-embedding-3-small.
# Đánh đổi: model ~2.3GB, lần chạy đầu tải khá lâu.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Vector store: ChromaDB — persistent local, không cần Docker, hỗ trợ cosine similarity
# sẵn qua `hnsw:space=cosine` (Weaviate cần Docker; FAISS không lưu metadata tiện như Chroma).
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"

# ChromaDB có giới hạn số record mỗi lần add/upsert — chia batch cho an toàn.
UPSERT_BATCH_SIZE = 200

# Chunk ngắn hơn ngưỡng này là rác do splitter cắt sót (dấu gạch ngang, số trang,
# ký tự lẻ). Embedding của chúng vô nghĩa và chỉ làm nhiễu kết quả retrieval.
MIN_CHUNK_CHARS = 40


# =============================================================================
# SHARED RESOURCES — dùng lại bởi Task 5 (semantic search)
# =============================================================================

_MODEL = None
_COLLECTION = None


def get_embedding_model():
    """
    Trả về SentenceTransformer đã cache.

    Singleton là bắt buộc: bge-m3 nặng ~2.3GB, load lại mỗi lần gọi semantic_search()
    sẽ khiến mỗi câu hỏi mất hàng chục giây.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        print(f"⏳ Đang load embedding model: {EMBEDDING_MODEL} ...")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
        print("✓ Model đã sẵn sàng")
    return _MODEL


def get_collection():
    """
    Trả về ChromaDB collection đã cache.

    `hnsw:space=cosine` → collection.query() trả về cosine DISTANCE trong [0, 2];
    Task 5 quy đổi sang similarity bằng `1 - distance`.
    """
    global _COLLECTION
    if _COLLECTION is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Chưa có {STANDARDIZED_DIR} — chạy Task 3 trước.")
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue

        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            # Key phải là "type" (không phải "doc_type") — app.py đọc meta.get("type").
            "metadata": {"source": md_file.name, "type": doc_type},
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (recursive, size=800, overlap=100).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ưu tiên cắt ở ranh giới heading → đoạn → dòng → câu, cuối cùng mới cắt giữa từ.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for chunk_text in splitter.split_text(doc["content"]):
            # ChromaDB báo lỗi nếu document là chuỗi rỗng; chunk vài ký tự thì tuy
            # hợp lệ nhưng embedding vô nghĩa — loại cả hai từ đây.
            if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": len(chunks)},
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]

    # normalize_embeddings=True: vector đơn vị → kết hợp với hnsw:space=cosine cho
    # distance ∈ [0, 2] và (1 - distance) đúng là cosine similarity ∈ [-1, 1].
    # Điều này cần thiết để ngưỡng SCORE_THRESHOLD ở Task 9 có ý nghĩa thật.
    embeddings = model.encode(
        texts,
        batch_size=8,          # bge-m3 khá nặng, batch nhỏ để không tràn RAM/VRAM
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB tại chroma_db/.

    Xoá collection cũ trước khi ghi để tránh chunk của corpus cũ còn sót lại
    (lỗi thường gặp #6 trong LAB_GUIDE).
    """
    global _COLLECTION

    if not chunks:
        print("⚠ Không có chunk nào để index.")
        return

    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  ✓ Đã xoá collection cũ '{COLLECTION_NAME}' để reindex sạch")
    except Exception:
        pass  # chưa tồn tại — lần chạy đầu tiên

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    _COLLECTION = collection

    # ID phải gồm cả 'type': legal/x.md và news/x.md có thể trùng tên file,
    # nếu chỉ dùng source thì chunk này sẽ ghi đè chunk kia.
    ids = [
        f"{c['metadata']['type']}_{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start:start + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=ids[start:start + UPSERT_BATCH_SIZE],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  ✓ Đã index {min(start + UPSERT_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} → {CHROMA_DIR}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")
    if chunks:
        sizes = [len(c["content"]) for c in chunks]
        print(f"  (kích thước chunk: min={min(sizes)}, max={max(sizes)}, "
              f"trung bình={sum(sizes) // len(sizes)})")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"\n✓ Hoàn tất — collection '{COLLECTION_NAME}' có {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
