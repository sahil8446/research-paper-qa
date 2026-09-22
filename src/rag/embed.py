"""Local, free embeddings via BGE-small (open-source, reproducible across machines)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# BGE models want a special instruction prefix on queries (not on the passages
# being indexed) -- it measurably improves retrieval for this model family.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_passages(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vectors = _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    vector = _model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
    return vector.tolist()
