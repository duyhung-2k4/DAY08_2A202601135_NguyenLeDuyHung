"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục

Chạy:
    python -m src.task3_convert_markdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Dưới ngưỡng này thì file .md gần như chắc chắn vô dụng cho retrieval
# (PDF bản scan, trang bị chặn bot, ...). Chỉ cảnh báo chứ không raise.
MIN_CONTENT_CHARS = 200


def convert_legal_docs() -> int:
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  ⚠ Chưa có data/landing/legal/ — chạy Task 1 trước.")
        return 0

    md = MarkItDown()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
        except Exception as exc:
            # Một PDF hỏng/scan không được phép làm chết cả lượt convert.
            print(f"  ✗ Lỗi convert: {type(exc).__name__}: {exc}")
            continue

        text = result.text_content or ""

        # KHÔNG ghi file rỗng: PDF bản scan (ảnh) trích ra 0 ký tự. Một file .md rỗng
        # vừa vô dụng cho retrieval, vừa làm hỏng test_converted_files_have_content
        # (test kiểm 5 file .md đầu tiên phải > 200 ký tự, mà legal/ đứng trước news/).
        if len(text) < MIN_CONTENT_CHARS:
            print(f"  ⚠ BỎ QUA {filepath.name}: chỉ trích được {len(text)} ký tự. "
                  f"Đây là PDF bản scan (ảnh) — MarkItDown không OCR được. "
                  f"Hãy thay bằng văn bản dạng text.")
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        converted += 1
        print(f"  ✓ Saved: {output_path.name} ({len(text):,} ký tự)")

    return converted


def convert_news_articles() -> int:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ Chưa có data/landing/news/ — chạy Task 2 trước.")
        return 0

    converted = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"  ✗ Lỗi đọc JSON: {exc}")
            continue

        # Header metadata được nhúng thẳng vào nội dung markdown, nên URL nguồn
        # sẽ đi theo vào từng chunk ở Task 4 → Task 10 có nguồn thật để trích dẫn.
        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = header + data.get("content_markdown", "")
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        converted += 1

        if len(content) < MIN_CONTENT_CHARS:
            print(f"  ⚠ Saved: {output_path.name} — chỉ {len(content)} ký tự!")
        else:
            print(f"  ✓ Saved: {output_path.name} ({len(content):,} ký tự)")

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 60)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 60)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print("\n" + "=" * 60)
    print(f"✓ Done! {n_legal} văn bản chính sách + {n_news} bài viết → {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    convert_all()
