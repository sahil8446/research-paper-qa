"""End-to-end baseline: embed the question, retrieve passages, generate a cited answer."""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from rag import config, vectorstore
from rag.embed import embed_query
from rag.llm import AnswerResult, generate_answer
from rag.schemas import Chunk


@dataclass
class Answered:
    result: AnswerResult
    sources: list[Chunk]  # same order as the [n] numbers cited in result.answer


def answer_question(
    question: str, client: QdrantClient | None = None, top_k: int = config.TOP_K
) -> Answered:
    client = client or vectorstore.get_client()
    sources = vectorstore.search(client, embed_query(question), top_k=top_k)
    result = generate_answer(question, sources)
    return Answered(result=result, sources=sources)
