"""Embed every chunk in data/processed/chunks.jsonl and load it into Qdrant."""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config, vectorstore  # noqa: E402
from rag.embed import embed_passages  # noqa: E402
from rag.schemas import Chunk  # noqa: E402


def load_chunks(path: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(x) for x in lines if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = ap.parse_args()

    chunks = load_chunks(config.CHUNKS_PATH)
    if not chunks:
        raise SystemExit(f"No chunks found at {config.CHUNKS_PATH}; run build_chunks.py first")

    client = vectorstore.get_client()
    vectorstore.ensure_collection(client, recreate=args.recreate)

    for i in tqdm(range(0, len(chunks), args.batch_size), desc="indexing"):
        batch = chunks[i : i + args.batch_size]
        vectors = embed_passages([c.text for c in batch], batch_size=args.batch_size)
        vectorstore.upsert_chunks(client, batch, vectors)

    count = client.count(config.QDRANT_COLLECTION).count
    store = config.QDRANT_URL or config.QDRANT_LOCAL_PATH
    print(f"indexed {len(chunks)} chunks -> {count} points in '{config.QDRANT_COLLECTION}'")
    print(f"store: {store}")


if __name__ == "__main__":
    main()
