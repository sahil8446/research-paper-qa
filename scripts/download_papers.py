"""Download arXiv papers on retrieval / RAG / agents and write data/manifest.jsonl.

Polite by design: one request at a time, ARXIV_DELAY_SECONDS between requests, and PDFs that
already exist on disk are skipped, so the script is safe to re-run.

Uses urllib rather than httpx: arXiv's API answered every httpx request with 406 Not Acceptable
(even with identical headers) while urllib works, so the stdlib client is the reliable choice here.
"""

import argparse
import http.client
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.schemas import PaperMeta  # noqa: E402

NS = {"a": "http://www.w3.org/2005/Atom"}
RETRY_STATUSES = {429, 500, 502, 503}


def parse_feed(xml_text: str) -> list[PaperMeta]:
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", NS):
        raw_id = entry.findtext("a:id", "", NS)
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        papers.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=" ".join(entry.findtext("a:title", "", NS).split()),
                authors=[n.findtext("a:name", "", NS) for n in entry.findall("a:author", NS)],
                year=int(entry.findtext("a:published", "0000", NS)[:4]),
                abstract=" ".join(entry.findtext("a:summary", "", NS).split()),
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


def fetch(url: str) -> tuple[int, bytes]:
    """GET with exponential backoff on throttling; returns (status, body)."""
    status, body = 0, b""
    for attempt in range(5):
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            status, body = err.code, b""
            if err.code not in RETRY_STATUSES:
                return status, body
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, OSError):
            status = 0  # dropped connection / timeout mid-download: worth a retry
        wait = config.ARXIV_DELAY_SECONDS * 2 ** (attempt + 1)
        tqdm.write(f"arXiv request failed (status {status}), retrying in {wait:.0f}s")
        time.sleep(wait)
    return status, body


def search(query: str, limit: int) -> list[PaperMeta]:
    cats = " OR ".join(f"cat:{c}" for c in config.SEARCH_CATEGORIES)
    q = quote_plus(f"({query}) AND ({cats})", safe=":()")
    url = f"{config.ARXIV_API}?search_query={q}&start=0&max_results={limit}&sortBy=relevance"
    status, body = fetch(url)
    if status != 200:
        raise RuntimeError(f"arXiv search failed with status {status} for: {query}")
    return parse_feed(body.decode("utf-8"))


def load_manifest() -> dict[str, PaperMeta]:
    if not config.MANIFEST_PATH.exists():
        return {}
    lines = config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    return {p.arxiv_id: p for p in (PaperMeta.model_validate_json(x) for x in lines if x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=400)
    ap.add_argument("--no-pdf", action="store_true", help="only build the manifest")
    args = ap.parse_args()

    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    per_query = max(10, -(-args.max_papers // len(config.SEARCH_QUERIES)) + 10)

    for query in config.SEARCH_QUERIES:
        if len(manifest) >= args.max_papers:
            break
        for paper in search(query, per_query):
            if len(manifest) < args.max_papers:
                manifest.setdefault(paper.arxiv_id, paper)
        time.sleep(config.ARXIV_DELAY_SECONDS)

    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(
        "\n".join(p.model_dump_json() for p in manifest.values()) + "\n", encoding="utf-8"
    )
    print(f"manifest: {len(manifest)} papers -> {config.MANIFEST_PATH}")

    if args.no_pdf:
        return
    for paper in tqdm(list(manifest.values()), desc="PDFs"):
        path = config.RAW_PDF_DIR / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        if path.exists():
            continue
        status, body = fetch(config.ARXIV_PDF.format(arxiv_id=paper.arxiv_id))
        if status == 200:
            path.write_bytes(body)
        else:
            tqdm.write(f"skip {paper.arxiv_id}: HTTP {status}")
        time.sleep(config.ARXIV_DELAY_SECONDS)


if __name__ == "__main__":
    main()
