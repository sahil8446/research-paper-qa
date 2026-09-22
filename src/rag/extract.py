"""PDF text extraction that keeps page numbers and section headings."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

NUMBER_PREFIX = re.compile(r"^(\d+(\.\d+)*|[IVX]+|[A-Z])\.?\s+")
BARE_NUMBER = re.compile(r"^(\d+(\.\d+)*|[IVX]+|[A-Z])\.?$")
KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "appendix",
}
SMALL_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with",
}  # fmt: skip


@dataclass
class Span:
    """A run of text on one page belonging to one section."""

    text: str
    page: int  # 1-based
    section: str


@dataclass
class Line:
    text: str
    page: int
    size: float
    bold: bool


def strip_number(heading: str) -> str:
    return NUMBER_PREFIX.sub("", heading.strip()).strip()


def _title_like(text: str) -> bool:
    """Headings are capitalised phrases; list items and sentence fragments are not."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]*", text) if w.lower() not in SMALL_WORDS]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) >= 0.6


def is_heading(text: str, bold: bool, larger: bool = False) -> bool:
    text = text.strip()
    if not text or len(text) > 90 or text.endswith((".", ",", ";")):
        return False
    if strip_number(text).lower() in KNOWN_HEADINGS:
        return True
    return bool(NUMBER_PREFIX.match(text)) and (bold or larger) and _title_like(strip_number(text))


def _read_lines(pdf_path: Path) -> list[Line]:
    lines: list[Line] = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  # skip images
                    continue
                for line in block["lines"]:
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if text:
                        first = line["spans"][0]
                        bold = any(s["flags"] & 16 for s in line["spans"])
                        lines.append(Line(text, page_no, first["size"], bold))
    return lines


def _body_size(lines: list[Line]) -> float:
    sizes = Counter()
    for ln in lines:
        sizes[round(ln.size, 1)] += len(ln.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def extract_spans(pdf_path: Path) -> list[Span]:
    """Return text spans tagged with page and current section heading."""
    lines = _read_lines(pdf_path)
    body = _body_size(lines)
    spans: list[Span] = []
    section = "Front matter"
    buffer: list[str] = []
    buffer_page = 1
    pending_number = ""  # ACM-style layouts put "2" and "Overview" on separate bold lines

    def flush() -> None:
        nonlocal buffer
        if buffer:
            spans.append(Span(" ".join(buffer), buffer_page, section))
            buffer = []

    for ln in lines:
        if ln.size < body * 0.8:  # figure / footnote text is never a heading or body text
            continue
        if pending_number:
            candidate = f"{pending_number} {ln.text}"
            pending_number = ""
            if ln.bold and len(ln.text) <= 90 and _title_like(ln.text):
                flush()
                section = candidate
                continue
        if ln.bold and BARE_NUMBER.match(ln.text):
            pending_number = ln.text.rstrip(".")
            continue
        if is_heading(ln.text, ln.bold, ln.size > body * 1.05):
            flush()
            section = ln.text
            continue
        if not buffer:
            buffer_page = ln.page
        elif ln.page != buffer_page:
            flush()
            buffer_page = ln.page
        buffer.append(ln.text)
    flush()
    return [_clean(s) for s in spans if s.text.strip()]


def _clean(span: Span) -> Span:
    text = re.sub(r"-\s+(?=[a-z])", "", span.text)  # re-join words hyphenated across lines
    text = re.sub(r"\s+", " ", text).strip()
    return Span(text, span.page, span.section)
