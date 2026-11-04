"""Document ingestion: load PDF/DOCX/TXT, clean text, and split into chunks.

Pipeline (pure functions: no network, no disk, no database):

    bytes -> validate_upload -> load_{pdf,docx,txt} -> clean -> chunk_pages -> list[Chunk]

The only public entry point most callers need is `ingest_file`.
Every expected failure is raised as `IngestionError` with a stable `code`, so the
API layer can map codes to HTTP statuses / Vietnamese messages without parsing text.
"""

import codecs
import hashlib
import io
import logging
import os
import re
import unicodedata
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import docx
import pymupdf
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lxml import etree

from app.config import Settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- constants

ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt")
MAX_FILENAME_LENGTH = 100

# A chunk shorter than this is noise (a stray page number, a lone heading) and would
# only pollute the vector index. Capped at chunk_size // 2 for tiny chunk sizes.
MIN_CHUNK_CHARS = 30

# A PDF with fewer extracted characters than this *per page on average* has no usable
# text layer (scanned/image-only). Real text pages have hundreds of characters.
MIN_PDF_CHARS_PER_PAGE = 25

# Basic zip-bomb guard for DOCX: refuse archives that would expand beyond this.
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024

# Fixed namespace => uuid5 is deterministic across runs and machines. Never change it
# after indexing, or every chunk_id changes and re-uploads would duplicate vectors.
CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "ai-learning-assistant.chunks")

# Coarse -> fine. The splitter tries the first separator that actually splits the text
# and only falls to the next one for pieces still larger than chunk_size.
# Regex lookbehind keeps the sentence punctuation attached to its own sentence.
SEPARATORS: list[str] = [
    r"\n\n",  # paragraphs
    r"\n",  # lines
    r"(?<=[.!?…])\s+",  # sentences
    r"(?<=[;:,])\s+",  # clauses
    r"\s+",  # words
    "",  # characters (last resort)
]

# Files that are definitely not text, whatever their extension claims.
_BINARY_SIGNATURES: tuple[bytes, ...] = (
    b"%PDF-",
    b"PK\x03\x04",  # zip / docx / xlsx
    b"\x89PNG",
    b"\xff\xd8\xff",  # jpeg
    b"GIF8",
    b"MZ",  # windows executable
    b"\xd0\xcf\x11\xe0",  # legacy OLE (.doc/.xls)
)

# --------------------------------------------------------------------------- types


class ErrorCode(StrEnum):
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    CONTENT_MISMATCH = "content_mismatch"  # extension says X, content is not X
    CORRUPTED_FILE = "corrupted_file"
    ENCRYPTED_PDF = "encrypted_pdf"
    SCANNED_PDF = "scanned_pdf"
    UNDECODABLE_TEXT = "undecodable_text"
    NO_TEXT = "no_text"


class IngestionError(Exception):
    """Expected, user-facing ingestion failure. `code` is stable; `message` is readable."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class PageText:
    page: int | None  # 1-based for PDF; None when the format has no reliable pages
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    filename: str  # sanitized
    extension: str  # lower-case, with the dot


@dataclass(frozen=True, slots=True)
class IngestResult:
    doc_id: str
    filename: str
    num_pages: int | None  # None for DOCX/TXT
    chunks: list[Chunk]


# --------------------------------------------------------------------------- validation

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\- ]+")  # \w is Unicode-aware: keeps Vietnamese


def sanitize_filename(filename: str) -> str:
    """Return a safe display name. (Files are stored under generated names, never this.)

    Strips any directory part (blocks `../../etc/passwd`), NFC-normalizes, replaces every
    character that is not a letter/digit/`_`/`.`/`-`/space (Unicode-aware, so Vietnamese
    letters survive) with `_`, and trims leading/trailing dots and spaces.
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = unicodedata.normalize("NFC", name)
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"_{2,}", "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    name = name.strip(" ._")
    if len(name) > MAX_FILENAME_LENGTH:
        stem, ext = os.path.splitext(name)
        keep = MAX_FILENAME_LENGTH - len(ext)
        name = stem[:keep] + ext if keep > 0 else name[:MAX_FILENAME_LENGTH]
    return name


def validate_upload(filename: str, data: bytes, max_mb: float) -> ValidatedUpload:
    """Check extension, emptiness, size and *real* content. Order: cheapest check first."""
    safe_name = sanitize_filename(filename)
    extension = os.path.splitext(safe_name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise IngestionError(
            ErrorCode.UNSUPPORTED_EXTENSION,
            f"Unsupported file type {extension or '(none)'!r}. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}.",
        )
    if not data:
        raise IngestionError(ErrorCode.EMPTY_FILE, "The file is empty.")
    max_bytes = int(max_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise IngestionError(
            ErrorCode.FILE_TOO_LARGE,
            f"The file is {len(data) / 1024 / 1024:.2f} MB; the limit is {max_mb:g} MB.",
        )
    _check_magic(extension, data)
    return ValidatedUpload(filename=safe_name, extension=extension)


def _check_magic(extension: str, data: bytes) -> None:
    """The extension is user-controlled; the first bytes are what the file really is."""
    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise IngestionError(
                ErrorCode.CONTENT_MISMATCH,
                "The file is named .pdf but its content is not a PDF document.",
            )
    elif extension == ".docx":
        _check_docx_container(data)
    elif _looks_binary(data):
        raise IngestionError(
            ErrorCode.CONTENT_MISMATCH,
            "The file is named .txt but its content is binary, not plain text.",
        )


def _check_docx_container(data: bytes) -> None:
    if not data.startswith(b"PK"):
        raise IngestionError(
            ErrorCode.CONTENT_MISMATCH,
            "The file is named .docx but is not a ZIP-based Word document.",
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise IngestionError(
            ErrorCode.CORRUPTED_FILE, "The DOCX archive is damaged or incomplete."
        ) from exc
    if "word/document.xml" not in {info.filename for info in infos}:
        raise IngestionError(
            ErrorCode.CONTENT_MISMATCH,
            "The file is a ZIP archive but not a Word document (word/document.xml is missing).",
        )
    if sum(info.file_size for info in infos) > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise IngestionError(
            ErrorCode.FILE_TOO_LARGE, "The DOCX expands to an unreasonable size when unpacked."
        )


def _looks_binary(data: bytes) -> bool:
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return False  # UTF-16 text legitimately contains NUL bytes
    return data.startswith(_BINARY_SIGNATURES) or b"\x00" in data[:8192]


# --------------------------------------------------------------------------- cleaning

_LINE_BREAK_CHARS = re.compile(r"[\x0b\x0c\x85  ]")  # vertical tab, form feed, NEL...
_INVISIBLE_CHARS = {ord(c): None for c in "­​⁠﻿"}  # soft hyphen, ZWSP, BOM
_UNICODE_HYPHENS = {ord("‐"): "-", ord("‑"): "-"}  # hyphen, non-breaking hyphen
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")  # all C0/C1 except \t \n
_HYPHEN_LINE_BREAK = re.compile(r"([^\W\d_])-[^\S\n]*\n[^\S\n]*([^\W\d_])")  # letter-\nletter
_INLINE_SPACE = re.compile(r"[^\S\n]+")  # any whitespace except newline (incl. NBSP, tabs)
_BLANK_LINES = re.compile(r"\n{3,}")


def _join_hyphenated(match: re.Match[str]) -> str:
    before, after = match.group(1), match.group(2)
    # Only join when the next word starts lowercase: "con-\ntainer" -> "container",
    # but "Client-\nServer" keeps its hyphen (a real compound).
    return before + after if after.islower() else match.group(0)


def clean_text(text: str) -> str:
    """Normalize raw extracted text. Idempotent. Rules, in order:

    1. NFC normalization. "ệ" can be one code point or e + 2 combining marks; without
       one canonical form the same word would embed/match differently.
    2. Unify line endings (\\r\\n, \\r, form feed, U+2028...) to "\\n".
    3. Delete invisible characters (soft hyphen, zero-width space, BOM) that break words,
       and map Unicode hyphens (U+2010, U+2011) to "-" so "B-tree" is one searchable form.
    4. Delete remaining control characters (keep \\n and \\t).
    5. Re-join words hyphenated across a line break (PDF layout artifact).
    6. Collapse runs of spaces/tabs/NBSP into one space (PyMuPDF emits NBSP often).
    7. Strip each line, and squeeze 3+ newlines into one blank line (keeps paragraphs).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_BREAK_CHARS.sub("\n", text)
    text = text.translate(_INVISIBLE_CHARS).translate(_UNICODE_HYPHENS)
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINE_BREAK.sub(_join_hyphenated, text)
    text = _INLINE_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Clean every page and drop the ones left empty (page numbers of the rest are kept)."""
    cleaned = [PageText(p.page, clean_text(p.text)) for p in pages]
    return [p for p in cleaned if p.text]


# --------------------------------------------------------------------------- loaders


def load_pdf(data: bytes) -> list[PageText]:
    """One PageText per PDF page (1-based), *including* empty ones, so numbering stays true."""
    pages: list[PageText] = []
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise IngestionError(
                    ErrorCode.ENCRYPTED_PDF, "The PDF is password-protected and cannot be read."
                )
            for number, page in enumerate(pdf, start=1):
                pages.append(PageText(number, _pdf_page_text(page)))
    except (RuntimeError, pymupdf.mupdf.FzErrorBase) as exc:  # FileDataError is a RuntimeError
        raise IngestionError(
            ErrorCode.CORRUPTED_FILE, "The PDF is damaged and could not be opened."
        ) from exc

    if not pages:
        raise IngestionError(ErrorCode.CORRUPTED_FILE, "The PDF contains no pages.")
    total_chars = sum(len("".join(p.text.split())) for p in pages)
    if total_chars < MIN_PDF_CHARS_PER_PAGE * len(pages):
        raise IngestionError(
            ErrorCode.SCANNED_PDF,
            "No text layer found: this looks like a scanned or image-only PDF. "
            "OCR is not supported; please upload a PDF with selectable text.",
        )
    empty = [p.page for p in pages if not p.text]
    if empty:
        logger.warning("PDF has %d page(s) without text (scanned?): %s", len(empty), empty[:20])
    return pages


def _pdf_page_text(page: "pymupdf.Page") -> str:
    """Text of one page as paragraphs separated by blank lines.

    PyMuPDF groups lines into blocks (usually = paragraphs, in reading order with
    sort=True). Inside a block the "\\n"s are just visual line wraps, so they become
    spaces; between blocks we keep a blank line for the splitter to prefer.
    """
    paragraphs: list[str] = []
    for _x0, _y0, _x1, _y1, block_text, _no, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:  # 0 = text, 1 = image
            continue
        # In a PDF text layer U+00AD is a *drawn* hyphen (many fonts map the hyphen glyph to
        # it), unlike in TXT/DOCX where it is an invisible hint. Keep it as "-" here; the
        # line-end rule in clean_text then re-joins real line-break hyphens.
        paragraph = clean_text(block_text.replace("­", "-")).replace("\n", " ")
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


# python-docx lets low-level errors escape on malformed packages (observed: BadZipFile,
# KeyError, TypeError, lxml XMLSyntaxError). The try block below is kept tiny for that reason.
_DOCX_ERRORS = (
    zipfile.BadZipFile,
    PackageNotFoundError,
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    etree.LxmlError,
)


def load_docx(data: bytes) -> list[PageText]:
    """Paragraphs and tables in document order. DOCX has no reliable pages -> page=None."""
    try:
        document = docx.Document(io.BytesIO(data))
        blocks = [
            _table_text(item) if isinstance(item, Table) else item.text
            for item in document.iter_inner_content()
            if isinstance(item, (Paragraph, Table))
        ]
    except _DOCX_ERRORS as exc:
        raise IngestionError(
            ErrorCode.CORRUPTED_FILE, "The DOCX is damaged and could not be read."
        ) from exc
    return [PageText(None, "\n\n".join(b for b in blocks if b.strip()))]


def _table_text(table: Table) -> str:
    """One line per row, cells joined by ' | '. Merged cells would repeat, so dedupe them."""
    lines: list[str] = []
    for row in table.rows:
        seen: set[object] = set()
        cells: list[str] = []
        for cell in row.cells:
            if cell._tc in seen:  # noqa: SLF001 - python-docx exposes no public identity
                continue
            seen.add(cell._tc)
            value = " ".join(cell.text.split())
            if value:
                cells.append(value)
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_txt(data: bytes) -> list[PageText]:
    return [PageText(None, _decode_text(data))]


def _decode_text(data: bytes) -> str:
    """utf-8-sig first (handles the Windows BOM); then UTF-16 (if BOM) and cp1258."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        has_utf16_bom = data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
        text = ""
        for encoding in ("utf-16", "cp1258") if has_utf16_bom else ("cp1258",):
            try:
                text = data.decode(encoding)
            except UnicodeDecodeError:
                continue
            logger.warning("Text file is not UTF-8; decoded with fallback encoding %s", encoding)
            break
        else:
            raise IngestionError(
                ErrorCode.UNDECODABLE_TEXT,
                "Could not decode the text file. Please save it as UTF-8 and upload again.",
            ) from None
    sample = text[:100_000]
    controls = sum(1 for ch in sample if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t\f")
    if controls > len(sample) * 0.01:
        raise IngestionError(
            ErrorCode.CONTENT_MISMATCH, "The file is named .txt but does not contain readable text."
        )
    return text


_LOADERS: dict[str, Callable[[bytes], list[PageText]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}

# --------------------------------------------------------------------------- ids + chunking


def compute_doc_id(data: bytes) -> str:
    """First 16 hex chars (64 bits) of SHA-256 of the file bytes: same file => same id."""
    return hashlib.sha256(data).hexdigest()[:16]


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID (Qdrant accepts UUID point ids) from document hash + position."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}:{chunk_index}"))


def chunk_pages(
    pages: list[PageText], doc_id: str, filename: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split each page on its own, so a chunk never straddles two pages.

    `keep_separator=True` is deliberate: with regex separators and keep_separator=False,
    langchain-text-splitters 1.1.3 leaks the regex source into the chunk text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        strip_whitespace=True,
    )
    min_chars = min(MIN_CHUNK_CHARS, chunk_size // 2)
    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            text = piece.strip()
            if len(text) < min_chars:
                continue
            index = len(chunks)  # global, contiguous; assigned after dropping tiny chunks
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, index),
                    doc_id=doc_id,
                    filename=filename,
                    page=page.page,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks


# --------------------------------------------------------------------------- public API


def ingest_file(filename: str, data: bytes, settings: Settings) -> IngestResult:
    """Validate, extract, clean and chunk one uploaded file. Raises IngestionError."""
    upload = validate_upload(filename, data, settings.max_upload_mb)
    doc_id = compute_doc_id(data)

    raw_pages = _LOADERS[upload.extension](data)
    num_pages = len(raw_pages) if upload.extension == ".pdf" else None
    pages = clean_pages(raw_pages)
    chunks = chunk_pages(
        pages, doc_id, upload.filename, settings.chunk_size, settings.chunk_overlap
    )
    if not chunks:
        raise IngestionError(
            ErrorCode.NO_TEXT, "No usable text could be extracted from the file."
        )
    logger.info(
        "Ingested %s: doc_id=%s pages=%s chunks=%d", upload.filename, doc_id, num_pages, len(chunks)
    )
    return IngestResult(doc_id=doc_id, filename=upload.filename, num_pages=num_pages, chunks=chunks)
