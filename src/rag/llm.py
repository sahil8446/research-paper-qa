"""Answer generation: Claude reads the retrieved passages and writes a cited answer.

Phase 2 baseline -- one call, no agent loop yet (that's Phase 5). The model is asked to
answer strictly from the given passages, cite them by number, and say so when the evidence
isn't there, so unanswerable questions (part of the Phase 3 eval set) get "not found" instead
of a guess.
"""

import json

import anthropic

from rag import config
from rag.schemas import Chunk

SYSTEM_PROMPT = """You are a research assistant answering questions using only the numbered \
passages provided below, drawn from a library of arXiv papers on retrieval, RAG and LLM agents.

Rules:
- Use only the passages given. Do not use outside knowledge.
- Every claim in your answer must be supported by at least one passage.
- Cite passages inline with their number in brackets, e.g. "RRF merges two ranked lists [2][4]."
- If the passages do not contain enough evidence to answer, set not_found to true, say briefly \
what is missing, and leave citations empty rather than guessing.
- citations must list every passage number you relied on, and only those."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "not_found": {"type": "boolean"},
    },
    "required": ["answer", "citations", "not_found"],
    "additionalProperties": False,
}


class AnswerResult:
    def __init__(self, answer: str, citations: list[int], not_found: bool):
        self.answer = answer
        self.citations = citations
        self.not_found = not_found


def format_passages(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        authors = ", ".join(c.authors[:3]) + (" et al." if len(c.authors) > 3 else "")
        pages = (
            f"p.{c.page_start}"
            if c.page_end == c.page_start
            else f"pp.{c.page_start}-{c.page_end}"
        )
        blocks.append(f"[{i}] {c.title} ({authors}, {c.year}) - {c.section}, {pages}\n{c.text}")
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    chunks: list[Chunk],
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
) -> AnswerResult:
    if not chunks:
        return AnswerResult(
            answer="No passages were retrieved for this question.", citations=[], not_found=True
        )

    client = client or anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model or config.ANSWER_MODEL,
            max_tokens=2048,  # a few cited sentences/paragraphs -- deliberately short answers
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": f"Passages:\n\n{format_passages(chunks)}\n\nQuestion: {question}",
                }
            ],
        )
    except anthropic.NotFoundError:
        raise RuntimeError(f"Unknown model '{model or config.ANSWER_MODEL}'") from None
    except anthropic.RateLimitError as err:
        retry_after = err.response.headers.get("retry-after", "unknown")
        raise RuntimeError(f"Claude API rate limited; retry after {retry_after}s") from err
    except anthropic.APIStatusError as err:
        raise RuntimeError(f"Claude API error ({err.status_code}): {err.message}") from err
    except anthropic.APIConnectionError as err:
        raise RuntimeError("Could not reach the Claude API -- check your connection") from err

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return AnswerResult(**data)
