"""Cut extracted spans into overlapping word-window chunks, one section at a time."""

from itertools import groupby

from rag import config
from rag.extract import Span, strip_number
from rag.schemas import Chunk, PaperMeta


def _section_key(name: str) -> str:
    return strip_number(name).lower()


def _windows(n: int, size: int, overlap: int, min_new: int) -> list[tuple[int, int]]:
    """Word index windows; a short tail is merged into the previous window."""
    step = size - overlap
    wins = [(s, min(s + size, n)) for s in range(0, n, step)]
    # drop windows fully covered by the previous one, merge short tails
    while len(wins) > 1 and wins[-1][1] - wins[-2][1] < min_new:
        wins.pop()
        wins[-1] = (wins[-1][0], n)
    return wins


def chunk_paper(
    meta: PaperMeta,
    spans: list[Span],
    size: int = config.CHUNK_WORDS,
    overlap: int = config.CHUNK_OVERLAP,
    min_new: int = config.MIN_NEW_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, group in groupby(spans, key=lambda s: s.section):
        if _section_key(section) in config.SKIP_SECTIONS:
            continue
        words: list[tuple[str, int]] = []
        for span in group:
            words.extend((w, span.page) for w in span.text.split())
        if len(words) < config.MIN_CHUNK_WORDS:
            continue
        for start, end in _windows(len(words), size, overlap, min_new):
            piece = words[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.arxiv_id}:{len(chunks)}",
                    arxiv_id=meta.arxiv_id,
                    title=meta.title,
                    authors=meta.authors,
                    year=meta.year,
                    section=section,
                    page_start=piece[0][1],
                    page_end=piece[-1][1],
                    text=" ".join(w for w, _ in piece),
                    n_words=len(piece),
                )
            )
    return chunks
