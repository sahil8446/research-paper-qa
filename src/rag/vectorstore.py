"""Qdrant wrapper. Points at a real server when QDRANT_URL is set (e.g. Docker), otherwise
falls back to an on-disk local collection -- same client API, so callers don't care which."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag import config
from rag.embed import EMBED_DIM
from rag.schemas import Chunk

# Qdrant point ids must be an int or a UUID; derive a stable UUID from chunk_id so re-running
# the indexer on the same chunks updates points in place instead of duplicating them.
ID_NAMESPACE = uuid.UUID("f3b2f6b0-2f39-4a8b-9b0a-7d6f6c0a6b6e")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_LOCAL_PATH)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    exists = client.collection_exists(config.QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    points = [
        qm.PointStruct(id=point_id(chunk.chunk_id), vector=vec, payload=chunk.model_dump())
        for chunk, vec in zip(chunks, vectors, strict=True)
    ]
    client.upsert(config.QDRANT_COLLECTION, points=points)


def search(
    client: QdrantClient, query_vector: list[float], top_k: int = config.TOP_K
) -> list[Chunk]:
    hits = client.query_points(config.QDRANT_COLLECTION, query=query_vector, limit=top_k).points
    return [Chunk.model_validate(hit.payload) for hit in hits]
