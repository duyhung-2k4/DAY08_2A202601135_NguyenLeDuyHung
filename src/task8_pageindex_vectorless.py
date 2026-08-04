"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fpdf import FPDF
from pageindex.client import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "pageindex_pdf"
DOC_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 300


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """PageIndex chỉ nhận PDF — convert markdown sang PDF tối giản bằng fpdf2."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    text = md_path.read_text(encoding="utf-8")
    for line in text.splitlines() or [""]:
        pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))
    pdf.output(str(pdf_path))


def _wait_until_retrieval_ready(client: PageIndexClient, doc_id: str) -> None:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        if client.is_retrieval_ready(doc_id):
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"PageIndex doc {doc_id} không sẵn sàng sau {POLL_TIMEOUT_SECONDS}s")


def upload_documents() -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Returns:
        dict: {md_filename: doc_id} — lưu lại vào data/pageindex_docs.json
        để pageindex_search() dùng lại mà không phải re-upload mỗi lần.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    registry: dict[str, str] = {}
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print("⚠ Không tìm thấy file .md nào trong data/standardized/ — chạy Task 3 trước.")
        return registry

    for md_file in md_files:
        pdf_path = PDF_CACHE_DIR / f"{md_file.stem}.pdf"
        _markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp["doc_id"]
        registry[md_file.name] = doc_id
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    DOC_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def _load_registry() -> dict[str, str]:
    if not DOC_REGISTRY_PATH.exists():
        return {}
    return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    registry = _load_registry()
    if not registry:
        raise RuntimeError(
            "Chưa có document nào trên PageIndex — chạy upload_documents() trước."
        )

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    results: list[dict] = []
    rank = 0
    for filename, doc_id in registry.items():
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp["retrieval_id"]

        deadline = time.time() + POLL_TIMEOUT_SECONDS
        retrieval = client.get_retrieval(retrieval_id)
        while retrieval.get("status") not in ("completed", "failed") and time.time() < deadline:
            time.sleep(POLL_INTERVAL_SECONDS)
            retrieval = client.get_retrieval(retrieval_id)

        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    rank += 1
                    results.append({
                        "content": item.get("relevant_content", ""),
                        # PageIndex không trả score trực tiếp — tự gán theo rank
                        # (giảm dần, tối đa 1.0) để nhất quán format với Task 5/6.
                        "score": max(0.0, 1.0 - 0.05 * (rank - 1)),
                        "metadata": {"source": filename, "section": item.get("section_title")},
                        "source": "pageindex",
                    })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
