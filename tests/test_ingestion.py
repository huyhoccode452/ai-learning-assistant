"""Tests for app/ingestion.py. All sample files are generated here; none are committed."""

import io
import logging
import re
import unicodedata
import uuid
import zipfile
from pathlib import Path

import docx
import pymupdf
import pytest

from app.config import Settings
from app.ingestion import (
    CHUNK_NAMESPACE,
    MIN_CHUNK_CHARS,
    Chunk,
    ErrorCode,
    IngestionError,
    PageText,
    chunk_pages,
    clean_text,
    ingest_file,
    sanitize_filename,
    validate_upload,
)

# --------------------------------------------------------------------------- helpers

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]

VI_SENTENCES = (
    "Cơ sở dữ liệu quan hệ lưu trữ thông tin dưới dạng bảng. "
    "Chỉ mục giúp tối ưu hóa truy vấn trên các bảng lớn. "
    "Phân vùng bảng làm cho việc bảo trì dữ liệu cũ trở nên dễ dàng hơn. "
)


def make_settings(**overrides: object) -> Settings:
    """Settings that ignore .env and the OS environment's chunking values."""
    values: dict[str, object] = {
        "gemini_api_key": "test-key-not-real",
        "chunk_size": 200,
        "chunk_overlap": 50,
        "max_upload_mb": 1,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def vi_font() -> str:
    """A TTF with Vietnamese glyphs. PyMuPDF's built-in fonts have none."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    pytest.skip("no Vietnamese-capable TTF font found on this machine")


def english_text(tag: str, sentences: int) -> str:
    return " ".join(f"{tag} sentence number {i:03d} talks about database indexes." for i in range(sentences))


def make_pdf(pages: list[str | None], font_file: str | None = None) -> bytes:
    """Build a PDF; a None entry is a blank page (no text, like a scan)."""
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text is None:
            continue
        fontname = "helv"
        if font_file:
            fontname = "vi"
            page.insert_font(fontname=fontname, fontfile=font_file)
        left = page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 790), text, fontsize=11, fontname=fontname
        )
        assert left >= 0, "test text does not fit on the page"
    return pdf.tobytes()


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    document = docx.Document()
    for paragraph in paragraphs[:1]:
        document.add_paragraph(paragraph)
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                grid.cell(r, c).text = value
    for paragraph in paragraphs[1:]:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def replace_zip_member(blob: bytes, member: str, content: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(blob)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, content if info.filename == member else src.read(info.filename))
    return out.getvalue()


def joined(chunks: list[Chunk]) -> str:
    return " ".join(c.text for c in chunks)


def assert_error(code: ErrorCode, filename: str, data: bytes, settings: Settings) -> IngestionError:
    with pytest.raises(IngestionError) as info:
        ingest_file(filename, data, settings)
    assert info.value.code == code, f"expected {code}, got {info.value.code}: {info.value.message}"
    assert info.value.message  # every error carries a readable message
    return info.value


# --------------------------------------------------------------------------- valid files


def test_valid_txt(settings: Settings) -> None:
    text = english_text("Alpha", 12)
    result = ingest_file("notes.txt", text.encode(), settings)
    assert result.filename == "notes.txt"
    assert result.num_pages is None
    assert len(result.chunks) > 1
    assert {c.page for c in result.chunks} == {None}
    assert [c.chunk_index for c in result.chunks] == list(range(len(result.chunks)))
    assert all(c.filename == "notes.txt" and c.doc_id == result.doc_id for c in result.chunks)
    assert all(len(c.text) <= settings.chunk_size for c in result.chunks)


def test_valid_docx_reads_paragraphs_and_tables_in_order() -> None:
    data = make_docx(
        ["Giới thiệu về cơ sở dữ liệu quan hệ.", "Kết luận cuối tài liệu."],
        table=[["Họ tên", "Điểm"], ["Nguyễn Văn An", "9.5"]],
    )
    result = ingest_file("bao_cao.docx", data, make_settings(chunk_size=800, chunk_overlap=100))
    assert result.num_pages is None
    assert len(result.chunks) == 1
    text = result.chunks[0].text
    assert result.chunks[0].page is None
    assert text.index("Giới thiệu") < text.index("Họ tên | Điểm") < text.index("Kết luận")
    assert "Nguyễn Văn An | 9.5" in text


def test_valid_pdf_has_one_based_page_numbers(settings: Settings) -> None:
    data = make_pdf([english_text("Alpha", 10), english_text("Bravo", 10)])
    result = ingest_file("doc.pdf", data, settings)
    assert result.num_pages == 2
    assert {c.page for c in result.chunks} == {1, 2}


# --------------------------------------------------------------------------- page numbers


def test_chunks_never_mix_pages_and_keep_correct_page_number() -> None:
    tags = {1: "ALPHA", 2: "BRAVO", 3: "CHARLIE"}
    data = make_pdf([english_text(tag, 10) for tag in tags.values()])
    result = ingest_file("three.pdf", data, make_settings())
    assert result.num_pages == 3
    for chunk in result.chunks:
        assert chunk.page is not None
        assert tags[chunk.page] in chunk.text
        assert not any(t in chunk.text for p, t in tags.items() if p != chunk.page)
    pages_in_order = [c.page for c in result.chunks]
    assert pages_in_order == sorted(pages_in_order)  # chunk_index follows page order


def test_blank_page_is_dropped_but_page_numbers_stay_true() -> None:
    data = make_pdf([english_text("First", 8), None, english_text("Third", 8)])
    result = ingest_file("gap.pdf", data, make_settings())
    assert result.num_pages == 3  # the blank page still counts
    assert {c.page for c in result.chunks} == {1, 3}  # ...but yields no chunk, and page 3 stays 3


# --------------------------------------------------------------------------- Vietnamese


def test_vietnamese_pdf_text_is_extracted_correctly(vi_font: str) -> None:
    data = make_pdf([VI_SENTENCES * 2], font_file=vi_font)  # embeds the whole TTF (~1 MB)
    result = ingest_file(
        "vi.pdf", data, make_settings(chunk_size=800, chunk_overlap=100, max_upload_mb=5)
    )
    text = joined(result.chunks)
    assert "tối ưu hóa truy vấn" in text
    assert "Phân vùng bảng" in text
    assert "\u00a0" not in text  # PyMuPDF emits NBSP for spaces; must be normalized
    assert unicodedata.is_normalized("NFC", text)


def test_pdf_keeps_visible_hyphens_in_technical_terms(vi_font: str) -> None:
    # Regression: PyMuPDF reports a visible "-" as U+00AD (soft hyphen) or U+2010 depending on
    # the font. Deleting soft hyphens blindly turned "B-tree" into "Btree".
    text = "Chỉ mục B-tree và Multi-Version Concurrency Control là nền tảng của PostgreSQL."
    data = make_pdf([text], font_file=vi_font)
    result = ingest_file(
        "hy.pdf", data, make_settings(chunk_size=800, chunk_overlap=100, max_upload_mb=5)
    )
    assert "B-tree" in result.chunks[0].text
    assert "Multi-Version" in result.chunks[0].text


def test_vietnamese_txt_utf8_with_bom_and_nfd_input_become_nfc() -> None:
    nfc = "Tiếng Việt là ngôn ngữ của người Việt Nam, rất giàu thanh điệu."
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc
    data = b"\xef\xbb\xbf" + nfd.encode("utf-8")  # UTF-8 BOM + decomposed form
    result = ingest_file("vi.txt", data, make_settings(chunk_size=800, chunk_overlap=100))
    assert [c.text for c in result.chunks] == [nfc]  # BOM gone, NFC restored


def test_txt_in_legacy_vietnamese_encoding_uses_fallback_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # cp1258 stores Vietnamese as base letter + combining mark (ệ = ê + U+0323).
    raw = "Ti\u00ea\u0301ng Vi\u00ea\u0323t l\u00e0 ng\u00f4n ng\u01b0\u0303 c\u0075\u0309a ng\u01b0\u01a1\u0300i Vi\u00ea\u0323t."
    expected = unicodedata.normalize("NFC", raw)
    assert expected == "Tiếng Việt là ngôn ngữ của người Việt."
    with caplog.at_level(logging.WARNING, logger="app.ingestion"):
        result = ingest_file("old.txt", raw.encode("cp1258"), make_settings(chunk_size=800))
    assert [c.text for c in result.chunks] == [expected]
    assert any("cp1258" in r.getMessage() for r in caplog.records)


def test_txt_utf16_with_bom_is_decoded(caplog: pytest.LogCaptureFixture) -> None:
    text = "Cơ sở dữ liệu quan hệ và chỉ mục B-tree trong PostgreSQL."
    with caplog.at_level(logging.WARNING, logger="app.ingestion"):
        result = ingest_file("u16.txt", text.encode("utf-16"), make_settings(chunk_size=800))
    assert [c.text for c in result.chunks] == [text]
    assert any("utf-16" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- chunking


def test_overlap_repeats_the_end_of_each_chunk_at_the_start_of_the_next() -> None:
    text = " ".join(f"Câu số {i:03d} nói về chỉ mục B-tree." for i in range(40))
    result = ingest_file("o.txt", text.encode(), make_settings(chunk_size=200, chunk_overlap=80))
    chunks = result.chunks
    assert len(chunks) > 3
    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = re.split(r"(?<=\.)\s+", current.text)[0]
        assert first_sentence in previous.text, "next chunk must start inside the previous one"


def test_zero_overlap_means_no_repeated_sentences() -> None:
    text = " ".join(f"Câu số {i:03d} nói về chỉ mục B-tree." for i in range(40))
    result = ingest_file("o.txt", text.encode(), make_settings(chunk_size=200, chunk_overlap=0))
    sentences = re.findall(r"Câu số \d{3}", joined(result.chunks))
    assert len(sentences) == len(set(sentences)) == 40


def test_splitter_does_not_leak_regex_source_into_chunks() -> None:
    # Regression: langchain-text-splitters 1.1.3 inserts the separator *regex* into the
    # text when is_separator_regex=True and keep_separator=False.
    # chunk_size must be large enough that several sentences are *merged* into one chunk;
    # the leak only happens at the merge step (a one-sentence-per-chunk setup hides it).
    pages = [PageText(1, english_text("Leak", 20))]
    chunks = chunk_pages(pages, "d" * 16, "f.pdf", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 3
    assert any(c.text.count("Leak sentence") >= 2 for c in chunks), "test must exercise merging"
    for chunk in chunks:
        assert "(?<=" not in chunk.text and "\\s" not in chunk.text and "\\n" not in chunk.text


def test_tiny_chunks_are_dropped_and_tiny_documents_are_rejected(settings: Settings) -> None:
    assert_error(ErrorCode.NO_TEXT, "tiny.txt", b"Xin chao.", settings)
    result = ingest_file("ok.txt", english_text("Z", 12).encode(), settings)
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in result.chunks)


# --------------------------------------------------------------------------- stable ids


def test_same_file_gives_identical_ids(settings: Settings) -> None:
    data = english_text("Same", 12).encode()
    first = ingest_file("a.txt", data, settings)
    second = ingest_file("a.txt", data, settings)
    assert first.doc_id == second.doc_id
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert len({c.chunk_id for c in first.chunks}) == len(first.chunks)  # unique within a doc


def test_renamed_copy_of_same_bytes_has_same_ids(settings: Settings) -> None:
    # Identity comes from content, not name: re-uploading under another name must not duplicate.
    data = english_text("Same", 12).encode()
    a = ingest_file("a.txt", data, settings)
    b = ingest_file("renamed.txt", data, settings)
    assert a.doc_id == b.doc_id
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]


def test_different_files_give_different_ids(settings: Settings) -> None:
    a = ingest_file("a.txt", english_text("Alpha", 12).encode(), settings)
    b = ingest_file("b.txt", english_text("Bravo", 12).encode(), settings)
    assert a.doc_id != b.doc_id
    assert not {c.chunk_id for c in a.chunks} & {c.chunk_id for c in b.chunks}


def test_chunk_id_formula_is_uuid5_of_doc_id_and_index(settings: Settings) -> None:
    result = ingest_file("a.txt", english_text("Alpha", 12).encode(), settings)
    for chunk in result.chunks:
        assert chunk.chunk_id == str(uuid.uuid5(CHUNK_NAMESPACE, f"{result.doc_id}:{chunk.chunk_index}"))
    assert len(result.doc_id) == 16


# --------------------------------------------------------------------------- rejected uploads


def test_empty_file(settings: Settings) -> None:
    assert_error(ErrorCode.EMPTY_FILE, "empty.txt", b"", settings)


def test_whitespace_only_txt_has_no_text(settings: Settings) -> None:
    assert_error(ErrorCode.NO_TEXT, "blank.txt", b"  \n\n \t \n", settings)


@pytest.mark.parametrize("name", ["malware.exe", "image.png", "notes.doc", "noextension", "x.PDF.exe"])
def test_wrong_extension(name: str, settings: Settings) -> None:
    assert_error(ErrorCode.UNSUPPORTED_EXTENSION, name, b"some content here", settings)


def test_extension_check_is_case_insensitive(settings: Settings) -> None:
    result = ingest_file("NOTES.TXT", english_text("Up", 10).encode(), settings)
    assert result.chunks


def test_fake_pdf_that_is_really_text(settings: Settings) -> None:
    assert_error(ErrorCode.CONTENT_MISMATCH, "fake.pdf", b"This is just plain text.", settings)


def test_fake_docx_that_is_really_text_or_a_plain_zip(settings: Settings) -> None:
    assert_error(ErrorCode.CONTENT_MISMATCH, "fake.docx", b"plain text, not a zip", settings)
    plain_zip = io.BytesIO()
    with zipfile.ZipFile(plain_zip, "w") as z:
        z.writestr("readme.txt", "hello")
    assert_error(ErrorCode.CONTENT_MISMATCH, "fake2.docx", plain_zip.getvalue(), settings)


def test_fake_txt_that_is_really_a_pdf_or_binary(settings: Settings) -> None:
    assert_error(ErrorCode.CONTENT_MISMATCH, "pdf_in_disguise.txt", make_pdf([english_text("X", 5)]), settings)
    assert_error(ErrorCode.CONTENT_MISMATCH, "image.txt", b"\x89PNG\r\n\x1a\n" + b"\x00" * 200, settings)
    assert_error(ErrorCode.CONTENT_MISMATCH, "random.txt", b"abc\x00def" * 100, settings)


def test_corrupted_pdf(settings: Settings) -> None:
    assert_error(ErrorCode.CORRUPTED_FILE, "bad.pdf", b"%PDF-1.7\n" + b"\x00garbage" * 80, settings)
    assert_error(ErrorCode.CORRUPTED_FILE, "bad2.pdf", b"%PDF-1.4", settings)


def test_truncated_pdf_raises_ingestion_error_not_a_library_error(settings: Settings) -> None:
    good = make_pdf([english_text("Cut", 10)])
    with pytest.raises(IngestionError):
        ingest_file("cut.pdf", good[: len(good) // 3], settings)


def test_corrupted_docx(settings: Settings) -> None:
    good = make_docx(["Một đoạn văn đủ dài để làm tài liệu kiểm thử."])
    # 1) truncated archive
    assert_error(ErrorCode.CORRUPTED_FILE, "trunc.docx", good[: len(good) // 2], settings)
    # 2) valid archive, but word/document.xml is not XML
    assert_error(
        ErrorCode.CORRUPTED_FILE,
        "badxml.docx",
        replace_zip_member(good, "word/document.xml", b"not xml <<<"),
        settings,
    )
    # 3) valid XML, wrong structure (python-docx raises TypeError here)
    empty_root = b"<?xml version='1.0'?><w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'/>"
    assert_error(
        ErrorCode.CORRUPTED_FILE, "root.docx", replace_zip_member(good, "word/document.xml", empty_root), settings
    )


def test_txt_with_undecodable_bytes(settings: Settings) -> None:
    # 0x81 is undefined in cp1258 and invalid in UTF-8: nothing can decode it.
    assert_error(ErrorCode.UNDECODABLE_TEXT, "weird.txt", b"abc \x81 def " * 20, settings)


def test_oversized_file(settings: Settings) -> None:
    one_mb = 1024 * 1024
    validate_upload("limit.txt", b"a" * one_mb, max_mb=1)  # exactly at the limit is allowed
    err = assert_error(ErrorCode.FILE_TOO_LARGE, "big.txt", b"a" * (one_mb + 1), settings)
    assert "1 MB" in err.message


def test_zip_bomb_like_docx_is_rejected(settings: Settings) -> None:
    good = make_docx(["Tài liệu kiểm thử."])
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"0" * (101 * 1024 * 1024))  # compresses to ~100 KB
    assert len(bomb.getvalue()) < 1024 * 1024  # passes the upload size limit...
    assert_error(ErrorCode.FILE_TOO_LARGE, "bomb.docx", bomb.getvalue(), settings)  # ...not the unpacked one
    assert good  # (sanity: the helper itself works)


# --------------------------------------------------------------------------- scanned / encrypted PDF


def test_blank_pdf_is_reported_as_scanned(settings: Settings) -> None:
    err = assert_error(ErrorCode.SCANNED_PDF, "scan.pdf", make_pdf([None]), settings)
    assert "OCR" in err.message
    assert_error(ErrorCode.SCANNED_PDF, "scan3.pdf", make_pdf([None, None, None]), settings)


def test_image_only_pdf_is_reported_as_scanned(settings: Settings) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 80))
    pixmap.set_rect(pixmap.irect, (200, 200, 200))
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pixmap)
    assert_error(ErrorCode.SCANNED_PDF, "photo.pdf", pdf.tobytes(), settings)


def test_encrypted_pdf(settings: Settings) -> None:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((50, 50), english_text("Secret", 5), fontsize=11)
    data = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user-pw", owner_pw="owner-pw")
    assert_error(ErrorCode.ENCRYPTED_PDF, "locked.pdf", data, settings)


# --------------------------------------------------------------------------- unit: clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Vie\u0323\u0302t", "Việt"),  # NFD -> NFC
        ("a  \t  b", "a b"),  # repeated spaces/tabs
        ("a\u00a0\u00a0b", "a b"),  # NBSP
        ("a\n\n\n\n\nb", "a\n\nb"),  # many blank lines -> one
        ("  line one  \n   line two   ", "line one\nline two"),  # per-line strip
        ("a\r\nb\rc", "a\nb\nc"),  # line endings
        ("a\x00b\x07c\x1fd", "abcd"),  # control chars
        ("a\u200bb\u00adc\ufeffd", "abcd"),  # zero-width space, soft hyphen, BOM
        ("B\u2010tree a\u2011b", "B-tree a-b"),  # Unicode hyphens -> ASCII hyphen
        ("con-\ntainer", "container"),  # hyphenated line break is re-joined
        ("con-  \n  tainer", "container"),  # ...even with stray spaces
        ("Client-\nServer", "Client-\nServer"),  # uppercase next word: real compound, keep
        ("năm 2024-\n2025", "năm 2024-\n2025"),  # digits: keep
        ("- mục một\n- mục hai", "- mục một\n- mục hai"),  # list bullets untouched
        ("  \n\n \t ", ""),  # only whitespace
    ],
)
def test_clean_text_rules(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent() -> None:
    messy = "Ti\u00ea\u0301ng  Vi\u00ea\u0323t\u00a0con-\ntainer\x00\n\n\n\nHết  "
    once = clean_text(messy)
    assert clean_text(once) == once


# --------------------------------------------------------------------------- unit: sanitize_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\sv\\Desktop\\bài tập.docx", "bài tập.docx"),
        ("báo cáo  năm 2025.pdf", "báo cáo năm 2025.pdf"),  # Vietnamese kept, spaces collapsed
        ('a<b>:c|d?"e*.txt', "a_b_c_d_e_.txt"),
        ("evil.pdf\x00.exe", "evil.pdf_.exe"),  # NUL-byte extension trick
        ("...hidden.txt", "hidden.txt"),
        ("name with trailing dot.txt.", "name with trailing dot.txt"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_truncates_but_keeps_extension() -> None:
    name = sanitize_filename("x" * 500 + ".pdf")
    assert len(name) == 100 and name.endswith(".pdf")


def test_nul_byte_extension_trick_is_rejected(settings: Settings) -> None:
    assert_error(ErrorCode.UNSUPPORTED_EXTENSION, "evil.pdf\x00.exe", b"%PDF-1.4 fake", settings)


def test_ingest_result_filename_is_sanitized(settings: Settings) -> None:
    result = ingest_file("../../tmp/my:notes.txt", english_text("S", 10).encode(), settings)
    assert result.filename == "my_notes.txt"
    assert all(c.filename == "my_notes.txt" for c in result.chunks)
