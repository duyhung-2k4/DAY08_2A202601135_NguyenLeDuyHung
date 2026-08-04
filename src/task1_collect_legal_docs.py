"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Chủ đề nhóm: **Đại học Quốc gia Hà Nội (VNU)** — vnu.edu.vn và các trường thành viên.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Nguồn sử dụng (trang công khai của ĐHQGHN và trường thành viên):
    - https://uet.vnu.edu.vn/category/sinh-vien/quy-dinh-quy-che/  (Quy chế, quy định sinh viên)
    - https://uet.vnu.edu.vn/quy-che-dao-tao-dai-hoc-tai-dhqghn/
    - https://uet.vnu.edu.vn/quy-dinh-ve-cong-tac-quan-ly-su-dung-hoc-bong-tai-dai-hoc-quoc-gia-ha-noi-2/

Chủ đề bao phủ (dịch vụ đại học):
    - Học phí & định mức thu (Tuition Fees)
    - Chính sách học bổng (Scholarship — QĐ 4618/QĐ-ĐHQGHN)
    - Quy chế đào tạo & đăng ký học phần (Course Registration)
    - Ký túc xá / hỗ trợ sinh viên (xem thêm Task 2 — css.vnu.edu.vn)

Lưu ý:
    - Một số trang trường chặn bot mặc định (HTTP 403) nếu request không có User-Agent —
      script này luôn gửi UA của trình duyệt thật.
    - Domain `tuyensinh.vnu.edu.vn` hiện có lỗi chuỗi chứng chỉ SSL ("unable to verify the
      first certificate") nên KHÔNG dùng làm nguồn ở đây.
    - Chỉ dùng nguồn công khai, được phép chia sẻ.

Chạy:
    python -m src.task1_collect_legal_docs
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Giả làm trình duyệt thật: WordPress + WAF của nhiều trường trả 403 với UA mặc định
# của requests ("python-requests/2.x").
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}

# Kích thước tối thiểu để coi là file hợp lệ. Trang lỗi/redirect thường trả về
# một trang HTML nhỏ — chặn ở đây để không lưu rác vào landing zone.
MIN_FILE_SIZE = 1024

# Tải 5 văn bản dù đề bài chỉ yêu cầu tối thiểu 3 — để còn đệm nếu 1-2 URL chết.
DOCUMENTS = [
    {
        "filename": "quy-che-dao-tao-dai-hoc-dhqghn-3626.pdf",
        # Dùng bản trên ussh.vnu.edu.vn (643 KB, PDF dạng text → trích được 95k ký tự).
        # KHÔNG dùng bản trên uet.edu.vn / media.isvnu.vn: cùng nội dung nhưng là bản
        # SCAN 18 MB, MarkItDown trích ra 0 ký tự nên vô dụng cho retrieval.
        "url": "https://ussh.vnu.edu.vn/vi/van-ban/detail/"
               "Quy-che-dao-tao-dai-hoc-tai-Dai-hoc-Quoc-gia-Ha-Noi-Ap-dung-tu-khoa-QH-2022-X-19452/"
               "?download=1&id=0",
        "desc": "Quy chế đào tạo đại học tại ĐHQGHN (QĐ 3626/QĐ-ĐHQGHN) — "
                "đăng ký học phần, đánh giá, xét tốt nghiệp",
    },
    {
        "filename": "quy-dinh-hoc-bong-dhqghn-4618.pdf",
        "url": "https://uet-test.uet.edu.vn/wp-content/uploads/2024/10/"
               "Signed.Signed.Signed.2024_10_4_QUY-DINH-VE-HOC-BONG-1.pdf",
        "desc": "Quy định quản lý, sử dụng học bổng tại ĐHQGHN (QĐ 4618/QĐ-ĐHQGHN, 07/10/2024)",
    },
    {
        "filename": "so-tay-hoc-vu-uet-vnu-2020.pdf",
        "url": "https://uet.vnu.edu.vn/wp-content/uploads/2020/08/"
               "S%E1%BB%95-tay-h%E1%BB%8Dc-v%E1%BB%A5.pdf",
        "desc": "Sổ tay học vụ UET-VNU 2020-2021 — học phí, học bổng, đăng ký học phần, ký túc xá",
    },
    {
        "filename": "so-tay-hoc-vu-uet-vnu-2019.pdf",
        "url": "https://uet.vnu.edu.vn/wp-content/uploads/2019/07/"
               "S%E1%BB%95-tay-h%E1%BB%8Dc-v%E1%BB%A5-2019.pdf",
        "desc": "Sổ tay học vụ UET-VNU 2019 — quy trình dịch vụ sinh viên",
    },
    {
        "filename": "quy-dinh-thoi-gian-hoc-tap-uet-vnu.pdf",
        "url": "https://uet.vnu.edu.vn/wp-content/uploads/2017/09/"
               "Cv-TKB-ch%C3%ADnh-th%E1%BB%A9c.pdf",
        "desc": "Quy định về thời gian học tập & thời khoá biểu của sinh viên",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> bool:
    """
    Tải một file về DATA_DIR. Trả về True nếu thành công.

    Không raise: một URL chết không được làm hỏng cả lượt chạy — chỉ cần
    tối thiểu 3/5 file tải được là đạt yêu cầu Task 1.
    """
    filepath = DATA_DIR / filename
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ✗ Lỗi tải {filename}: {exc}")
        return False

    if len(response.content) < MIN_FILE_SIZE:
        print(f"  ✗ Bỏ qua {filename}: file quá nhỏ ({len(response.content)} bytes) "
              f"— nhiều khả năng là trang lỗi chứ không phải PDF")
        return False

    filepath.write_bytes(response.content)
    print(f"  ✓ Đã tải: {filepath.name} ({len(response.content) / 1024:.0f} KB)")
    return True


def collect_all() -> int:
    """Tải toàn bộ văn bản trong DOCUMENTS. Trả về số file tải thành công."""
    print("=" * 60)
    print("Task 1: Thu thập văn bản chính sách — ĐHQGHN (VNU)")
    print("=" * 60)
    setup_directory()

    success = 0
    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"\n[{i}/{len(DOCUMENTS)}] {doc['desc']}")
        if download_file(doc["url"], doc["filename"]):
            success += 1

    print("\n" + "=" * 60)
    print(f"Kết quả: {success}/{len(DOCUMENTS)} file tải thành công")
    if success < 3:
        print("⚠ CẢNH BÁO: cần tối thiểu 3 file để đạt Task 1.")
        print("  Tìm link thay thế tại: https://uet.vnu.edu.vn/category/sinh-vien/quy-dinh-quy-che/")
    else:
        print("✓ Đã đạt yêu cầu tối thiểu 3 văn bản chính sách.")
    print("=" * 60)
    return success


if __name__ == "__main__":
    collect_all()
