# Research Paper Q&A Assistant (Agentic RAG with Evaluation)

An assistant that answers questions about research papers and cites the exact paper and passage
behind every answer, plus an evaluation harness that measures whether each improvement helps.

> Status: **Phase 2 (baseline) done.** 355 arXiv papers, 9,694 chunks, embedded with BGE-small
> and indexed in Qdrant. Meaning search + Claude answer generation with citations, in a small
> Streamlit demo. Phase 3 (the 100-question eval set and scoring) is next.

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

cp .env.example .env   # fill in ANTHROPIC_API_KEY
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
python scripts/build_index.py                          # embeds all chunks into Qdrant, ~45 min on CPU

streamlit run app.py                                    # http://localhost:8501
pytest
```

`download_papers.py` is polite to arXiv on purpose: one request at a time, a few seconds apart,
with retries on throttling. Downloading the full library takes on the order of an hour.

PDFs and the generated chunks are not committed (they're large and reproducible). `manifest.jsonl`
*is* committed: it's small and records each paper's arXiv id, title and authors, so the exact same
355-paper library can be re-downloaded without re-running the arXiv searches.

Qdrant needs no server at all if `QDRANT_URL` is left unset in `.env` -- it falls back to a local
on-disk collection (`data/qdrant_local/`, also gitignored). Set `QDRANT_URL=http://localhost:6333`
once the Docker container above is running; same code path either way.

## Layout

```
src/rag/        config, schemas, PDF extraction, chunking, embeddings, vector store, answer generation
scripts/        download_papers.py, build_chunks.py, build_index.py
app.py          Streamlit demo
tests/          unit tests
data/           raw_pdfs/, processed/, qdrant_local/ (all gitignored)
```
