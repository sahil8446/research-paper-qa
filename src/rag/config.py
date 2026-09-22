from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# Chunking (chunk size is an experiment later, so keep it configurable)
CHUNK_WORDS = 400
CHUNK_OVERLAP = 60
MIN_CHUNK_WORDS = 20  # sections shorter than this are extraction debris (stray captions etc.)
MIN_NEW_WORDS = 80 # a trailing window with fewer new words is merged into the previous chunk

# arXiv asks automated clients to wait ~3 seconds between requests
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_DELAY_SECONDS = 3.0
USER_AGENT = "research-paper-qa/0.1 (student project)"

# Sections that add noise to retrieval
SKIP_SECTIONS = {"references", "bibliography", "acknowledgments", "acknowledgements"}

SEARCH_QUERIES = [
    'abs:"retrieval-augmented generation"',
    'abs:"dense retrieval"',
    'abs:"reranking" AND abs:"retrieval"',
    'abs:"LLM agents"',
    'abs:"tool use" AND abs:"language models"',
    'abs:"query rewriting"',
    'abs:"long context" AND abs:"retrieval"',
    'abs:"hallucination" AND abs:"retrieval"',
]
SEARCH_CATEGORIES = ["cs.CL", "cs.IR", "cs.AI", "cs.LG"]
