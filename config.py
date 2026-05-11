# ─────────────────────────────────────────────
# config.py — Central configuration
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ──────────────────────────────
# Choose: "ollama" (free local) | "openai" | "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL    = os.getenv("LLM_MODEL", "gemma4")        # ollama model name
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY= os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = 180   # seconds — add this line

# ── Embedding Settings ────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Free HuggingFace model, 384-dim
EMBEDDING_DIM   = 384

# ── Vector DB Settings ────────────────────────
FAISS_INDEX_PATH = "data/faiss_index"
METADATA_PATH    = "data/metadata.json"

# ── Chunking Settings ─────────────────────────
CHUNK_SIZE    = 400    # words per chunk
CHUNK_OVERLAP = 50     # overlap words between chunks

# ── Retrieval Settings ────────────────────────
TOP_K              = 3      # number of chunks to retrieve
CONFIDENCE_THRESHOLD = 0.40  # below this → escalate to human

# ── Evaluation Settings ───────────────────────
EVAL_DATASET_PATH = "data/eval_dataset.json"

# ── Data Paths ────────────────────────────────
DOCS_DIR    = "data/documents"
TICKETS_CSV = "data/tickets.csv"

# ── UI Settings ───────────────────────────────
APP_TITLE = "Enterprise Knowledge Assistant"
APP_ICON  = "🧠"
