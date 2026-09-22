from pydantic import BaseModel


class PaperMeta(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str = ""
    url: str


class Chunk(BaseModel):
    chunk_id: str  # "{arxiv_id}:{index}"
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    section: str
    page_start: int  # 1-based
    page_end: int
    text: str
    n_words: int
