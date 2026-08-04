"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Chủ đề nhóm: **Đại học Quốc gia Hà Nội (VNU)**.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Nếu KHÔNG cài được crawl4ai/playwright (máy yếu, mạng chậm, hoặc lab chỉ có 180 phút),
module này tự động rơi xuống `crawl_article_fallback()` dùng `requests` + `MarkItDown` —
cả hai đều đã có sẵn trong requirements.txt nên không cần thêm dependency nào.

Nguồn crawl (trang công khai của ĐHQGHN):
    - css.vnu.edu.vn — Trung tâm Hỗ trợ sinh viên (ký túc xá, nội trú, việc làm)
    - uet.vnu.edu.vn — Trường ĐH Công nghệ (học phí, học bổng, quy chế đào tạo)

Chạy:
    python -m src.task2_crawl_news
"""

import asyncio
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
}


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# 8 bài viết (yêu cầu tối thiểu 5) — bao phủ đủ 4 nhóm dịch vụ đại học:
# ký túc xá / nội trú, hỗ trợ sinh viên & việc làm, học phí, học bổng & quy chế đào tạo.
ARTICLE_URLS = [
    # --- Ký túc xá & nội trú (Trung tâm Hỗ trợ sinh viên ĐHQGHN) ---
    "https://css.vnu.edu.vn/thu-tuc-dang-ky-ky-tuc-xa-dhqghn/",
    "https://css.vnu.edu.vn/ky-tuc-xa/",
    "https://css.vnu.edu.vn/gioi-thieu-ve-ktx-hoa-lac-khu-bc/",
    "https://css.vnu.edu.vn/cong-bo-danh-sach-xet-duyet-va-huong-dan-lam-thu-tuc-noi-tru-"
    "nam-hoc-2025-2026-cho-doi-tuong-hoc-sinh-chuyen/",
    "https://css.vnu.edu.vn/trung-tam-ho-tro-sinh-vien-tiep-nhan-cong-trinh-nha-d1-d7-d8-"
    "san-sang-cac-dieu-kien-phuc-vu-sinh-vien-tai-hoa-lac/",
    # --- Hỗ trợ sinh viên & việc làm ---
    "https://css.vnu.edu.vn/tim-viec-lam-thuc-tap-de-dang-hon-cung-cong-thong-tin-viec-lam-"
    "sinh-vien-dhqghn/",
    # --- Học phí, học bổng, quy chế đào tạo (Trường ĐH Công nghệ) ---
    "https://uet.vnu.edu.vn/dinh-muc-hoc-phi-cac-chuong-trinh-dao-tao-nam-hoc-2025-2026/",
    "https://uet.vnu.edu.vn/quy-che-dao-tao-dai-hoc-tai-dhqghn/",
    "https://uet.vnu.edu.vn/quy-dinh-ve-cong-tac-quan-ly-su-dung-hoc-bong-tai-dai-hoc-"
    "quoc-gia-ha-noi-2/",
]


# --- Lọc boilerplate ----------------------------------------------------------
# Trang WordPress của các trường (css.vnu.edu.vn, uet.vnu.edu.vn) có menu điều hướng,
# breadcrumb và "Danh sách chuyên mục" khổng lồ: đo thực tế ~85% ký tự của mỗi trang là
# nav chứ không phải nội dung. Nếu index thẳng, phần lớn chunk sẽ là danh sách link vô
# nghĩa và đẩy nội dung thật ra khỏi top-k.

# Dòng dạng "* [Giới thiệu](https://...)" — mục menu, không phải nội dung.
_LINK_ONLY_LINE = re.compile(r"^\s*[*+\-]?\s*!?\[[^\]]*\]\([^)]*\)\s*$")
# Dòng chỉ chứa ảnh.
_IMAGE_ONLY_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def clean_markdown(text: str) -> str:
    """
    Loại bỏ menu điều hướng / breadcrumb khỏi markdown crawl được.

    Heuristic dựa trên link-density: một dòng mà phần lớn ký tự nằm trong markdown link
    thì gần như chắc chắn là mục menu chứ không phải câu văn.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue

        if _LINK_ONLY_LINE.match(stripped) or _IMAGE_ONLY_LINE.match(stripped):
            continue

        # Link-density: so phần text nằm ngoài link với tổng độ dài.
        without_links = _MD_LINK.sub(r"\1", stripped)
        link_chars = len(stripped) - len(without_links)
        if len(stripped) > 0 and link_chars / len(stripped) > 0.5:
            continue

        kept.append(without_links if link_chars else stripped)

    # Gộp các dòng trống liên tiếp.
    out = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def strip_shared_boilerplate(articles: list[dict], min_share: float = 0.6) -> list[dict]:
    """
    Xoá các dòng xuất hiện ở hầu hết bài viết (header/footer/sidebar dùng chung template).

    Đây là kỹ thuật boilerplate removal kinh điển: nội dung riêng của một bài chỉ xuất
    hiện ở bài đó, còn khung template lặp lại trên mọi trang cùng site.
    """
    if len(articles) < 3:
        return articles

    from collections import Counter

    counter = Counter()
    for a in articles:
        # set() để một dòng lặp nhiều lần trong CÙNG bài chỉ tính 1 lần
        counter.update({ln.strip() for ln in a["content_markdown"].splitlines() if ln.strip()})

    threshold = max(3, int(len(articles) * min_share))
    common = {ln for ln, n in counter.items() if n >= threshold}

    for a in articles:
        a["content_markdown"] = re.sub(
            r"\n{3,}",
            "\n\n",
            "\n".join(
                ln for ln in a["content_markdown"].splitlines()
                if ln.strip() not in common
            ),
        ).strip()

    return articles


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết bằng Crawl4AI và trả về dict chứa metadata + content.

    Dùng PruningContentFilter → `fit_markdown`: Crawl4AI chấm điểm từng khối DOM theo
    mật độ text/link rồi cắt bỏ khối có điểm thấp (menu, sidebar, footer), nên nội dung
    thu được sạch hơn hẳn raw_markdown.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.45, threshold_type="dynamic"),
        ),
        excluded_tags=["nav", "header", "footer", "aside", "form"],
        exclude_external_links=True,
    )

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url, config=config)

        # crawl4ai >= 0.4 trả về MarkdownGenerationResult (không phải str) ở result.markdown.
        # json.dumps sẽ ném TypeError nếu không ép về str trước.
        md_obj = result.markdown
        markdown = (
            getattr(md_obj, "fit_markdown", "")
            or getattr(md_obj, "raw_markdown", "")
            or str(md_obj)
        )

        return {
            "url": url,
            "title": (result.metadata or {}).get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": clean_markdown(str(markdown)),
        }


def crawl_article_fallback(url: str) -> dict:
    """
    Fallback không cần Playwright: tải HTML bằng requests rồi để MarkItDown
    chuyển sang Markdown.

    Dùng khi crawl4ai chưa cài hoặc `playwright install chromium` chưa chạy.
    Trả về đúng contract như crawl_article().
    """
    from markitdown import MarkItDown

    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    # requests đoán encoding từ header; trang tiếng Việt thường là utf-8 nhưng
    # nhiều server không khai báo charset → ép về utf-8 để tránh chữ bị hỏng.
    response.encoding = response.apparent_encoding or "utf-8"

    # MarkItDown đọc từ file nên phải ghi HTML ra file tạm trước.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_html = Path(tmpdir) / "page.html"
        tmp_html.write_text(response.text, encoding="utf-8")
        result = MarkItDown().convert(str(tmp_html))

    return {
        "url": url,
        "title": (result.title or "Unknown").strip(),
        "date_crawled": datetime.now().isoformat(),
        # MarkItDown giữ nguyên toàn bộ menu điều hướng → phải tự lọc, nếu không
        # ~85% nội dung là link rác.
        "content_markdown": clean_markdown(result.text_content or ""),
    }


async def fetch_one(url: str) -> dict:
    """Thử Crawl4AI trước, lỗi thì rơi xuống fallback requests + MarkItDown."""
    try:
        return await crawl_article(url)
    except Exception as exc:  # ImportError, browser chưa cài, timeout, ...
        print(f"  ⚠ Crawl4AI không dùng được ({type(exc).__name__}) → fallback MarkItDown")
        return crawl_article_fallback(url)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    articles = []
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await fetch_one(url)
        except Exception as exc:
            print(f"  ✗ Bỏ qua (lỗi: {exc})")
            continue

        if len(article["content_markdown"].strip()) < 200:
            print("  ✗ Bỏ qua: nội dung quá ngắn, nhiều khả năng trang chặn bot hoặc rỗng")
            continue

        articles.append(article)
        print(f"  ✓ Crawled ({len(article['content_markdown']):,} ký tự)")

    # Chạy sau khi có đủ bài: cần so sánh chéo giữa các bài mới phát hiện được
    # phần template dùng chung.
    before = sum(len(a["content_markdown"]) for a in articles)
    articles = strip_shared_boilerplate(articles)
    after = sum(len(a["content_markdown"]) for a in articles)
    if before:
        print(f"\n✓ Đã loại boilerplate dùng chung: {before:,} → {after:,} ký tự "
              f"(giảm {100 * (before - after) / before:.0f}%)")

    saved = 0
    for article in articles:
        if len(article["content_markdown"].strip()) < 200:
            print(f"  ✗ Bỏ {article['url']}: sau khi lọc còn quá ngắn")
            continue
        # Đánh số theo số file đã lưu (không theo index URL) để dãy article_NN.json liền mạch.
        saved += 1
        filepath = DATA_DIR / f"article_{saved:02d}.json"
        # encoding="utf-8" là BẮT BUỘC: mặc định của write_text trên Windows là cp1252,
        # gặp tiếng Việt sẽ ném UnicodeEncodeError.
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath.name} ({len(article['content_markdown']):,} ký tự)")

    print("\n" + "=" * 60)
    print(f"Kết quả: {saved}/{len(ARTICLE_URLS)} bài crawl thành công")
    if saved < 5:
        print("⚠ CẢNH BÁO: cần tối thiểu 5 bài để đạt Task 2.")
    else:
        print("✓ Đã đạt yêu cầu tối thiểu 5 bài viết.")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Task 2: Crawl bài viết dịch vụ sinh viên — ĐHQGHN (VNU)")
    print("=" * 60)
    asyncio.run(crawl_all())
