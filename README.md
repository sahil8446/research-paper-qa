# Research Paper Q&A Assistant (Agentic RAG with Evaluation)

An assistant that answers questions about research papers and cites the exact paper and passage
behind every answer, plus an evaluation harness that measures whether each improvement helps.

> Status: **Phase 1 (setup and data) done.** 355 arXiv papers on retrieval, RAG and LLM agents,
> chunked into 9,694 passages with page-level citations. Phase 2 (embeddings, search, baseline
> answering) is next.

## Results

| Version | Right passage found | Answer correct | Claims backed | Seconds per question |
| --- | --- | --- | --- | --- |
| Keyword search only | to measure | to measure | to measure | to measure |
| Meaning search only | to measure | to measure | to measure | to measure |
| Hybrid | to measure | to measure | to measure | to measure |
| Hybrid + reranker | to measure | to measure | to measure | to measure |
| + question rewriting | to measure | to measure | to measure | to measure |
| + agent loop and self-check | to measure | to measure | to measure | to measure |

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/download_papers.py --max-papers 400     # PDFs + data/manifest.jsonl
python scripts/build_chunks.py                         # data/processed/chunks.jsonl
pytest
```

`download_papers.py` is polite to arXiv on purpose: one request at a time, a few seconds apart,
with retries on throttling. Downloading the full library takes on the order of an hour.

PDFs and the generated chunks are not committed (they're large and reproducible). `manifest.jsonl`
*is* committed: it's small and records each paper's arXiv id, title and authors, so the exact same
355-paper library can be re-downloaded without re-running the arXiv searches.

## Layout

```
src/rag/        config, schemas, PDF extraction, chunking
scripts/        download_papers.py, build_chunks.py
tests/          unit tests
data/           raw_pdfs/ and processed/ (gitignored)
```
