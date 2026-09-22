from rag.chunking import _windows, chunk_paper
from rag.extract import Span, is_heading
from rag.schemas import PaperMeta

META = PaperMeta(
    arxiv_id="2401.00001", title="T", authors=["A"], year=2024, url="https://arxiv.org/abs/2401.00001"
)


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_windows_cover_everything_with_overlap():
    wins = _windows(1000, 400, 60, 80)
    assert wins[0][0] == 0 and wins[-1][1] == 1000
    for (_, prev_end), (start, _) in zip(wins, wins[1:], strict=False):
        assert prev_end - start == 60


def test_short_tail_is_merged():
    # a second window would add only 30 new words (400 -> 430), so it is merged into the first
    assert _windows(430, 400, 60, 80) == [(0, 430)]
    # enough new words: keep the second window
    assert _windows(700, 400, 60, 80) == [(0, 400), (340, 700)]


def test_short_section_single_chunk():
    chunks = chunk_paper(META, [Span(words(50), 1, "Abstract")])
    assert len(chunks) == 1 and chunks[0].n_words == 50


def test_debris_sections_are_dropped():
    assert chunk_paper(META, [Span("From the empirical", 6, "4.2.1 Main Results")]) == []


def test_chunks_carry_page_and_section_and_ids():
    spans = [Span(words(300), 1, "1 Introduction"), Span(words(300), 2, "1 Introduction")]
    chunks = chunk_paper(META, spans)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    assert {c.section for c in chunks} == {"1 Introduction"}
    assert [c.chunk_id for c in chunks] == [f"2401.00001:{i}" for i in range(len(chunks))]


def test_references_are_skipped():
    spans = [Span(words(100), 1, "Introduction"), Span(words(100), 5, "References")]
    assert {c.section for c in chunk_paper(META, spans)} == {"Introduction"}


def test_heading_detection():
    assert is_heading("Introduction", False)
    assert is_heading("3.1 Hybrid Retrieval", True)
    assert not is_heading("3.1 Hybrid Retrieval", False)
    assert not is_heading("We propose a method that improves retrieval quality.", True)
    assert is_heading("II. RELATED WORK", False)
    # bold list item from a figure: not title-like, so not a heading
    assert not is_heading("1. Identify the evidence relevant to", True)
