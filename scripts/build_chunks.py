"""Extract text from downloaded PDFs, chunk it, and write data/processed/chunks.jsonl."""

import argparse
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.chunking import chunk_paper  # noqa: E402
from rag.extract import extract_spans  # noqa: E402
from rag.schemas import PaperMeta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--out", type=Path, default=config.CHUNKS_PATH)
    args = ap.parse_args()

    lines = config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    papers = [PaperMeta.model_validate_json(x) for x in lines if x]

    total, failed, empty, sizes = 0, [], [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for paper in tqdm(papers, desc="chunking"):
            pdf = config.RAW_PDF_DIR / f"{paper.arxiv_id.replace('/', '_')}.pdf"
            if not pdf.exists():
                continue
            try:
                chunks = chunk_paper(paper, extract_spans(pdf), args.chunk_words, args.overlap)
            except Exception as exc:  # corrupt PDF etc.; keep going, report at the end
                failed.append((paper.arxiv_id, str(exc)))
                continue
            if not chunks:
                empty.append(paper.arxiv_id)
            for c in chunks:
                fh.write(c.model_dump_json() + "\n")
                sizes.append(c.n_words)
            total += len(chunks)

    print(f"\n{total} chunks -> {args.out}")
    if sizes:
        median = statistics.median(sizes)
        print(f"words/chunk: median {median:.0f}, min {min(sizes)}, max {max(sizes)}")
    print(f"papers with no text extracted: {len(empty)} {empty[:5]}")
    print(f"papers failed: {len(failed)} {failed[:5]}")


if __name__ == "__main__":
    main()
