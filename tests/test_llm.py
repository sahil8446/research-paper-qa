from rag.llm import format_passages, generate_answer
from rag.schemas import Chunk

CHUNK = Chunk(
    chunk_id="2401.00001:0",
    arxiv_id="2401.00001",
    title="Reciprocal Rank Fusion for Hybrid Search",
    authors=["A. One", "B. Two", "C. Three", "D. Four"],
    year=2024,
    section="3 Method",
    page_start=4,
    page_end=5,
    text="RRF merges ranked lists by summing 1/(k+rank) across each list.",
    n_words=11,
)


def test_format_passages_numbers_and_page_range():
    out = format_passages([CHUNK])
    assert out.startswith("[1] Reciprocal Rank Fusion for Hybrid Search")
    assert "A. One, B. Two, C. Three et al." in out
    assert "pp.4-5" in out
    assert CHUNK.text in out


def test_format_passages_single_page_no_range():
    single = CHUNK.model_copy(update={"page_end": CHUNK.page_start})
    assert "p.4" in format_passages([single])
    assert "pp." not in format_passages([single])


def test_generate_answer_short_circuits_with_no_chunks():
    result = generate_answer("Does longer context always help?", [])
    assert result.not_found is True
    assert result.citations == []
