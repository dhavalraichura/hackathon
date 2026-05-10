# ─────────────────────────────────────────────
# ingest.py — Data ingestion pipeline
# Handles: PDFs, HTML docs, CSV tickets, plain text
# ─────────────────────────────────────────────

import os
import json
import re
import time
import pandas as pd
import numpy as np
import faiss
import requests
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from bs4 import BeautifulSoup

import config

# ── Helpers ───────────────────────────────────

def clean_text(text: str) -> str:
    """Remove noise from raw text."""
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-ASCII
    text = text.strip()
    return text


def chunk_text(text: str, source: str, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping word chunks."""
    words  = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text  = " ".join(chunk_words)
        if len(chunk_words) > 30:   # skip tiny chunks
            chunks.append({
                "id":     f"{source}__chunk_{chunk_id}",
                "source": source,
                "text":   chunk_text,
            })
            chunk_id += 1
        i += chunk_size - overlap
    return chunks


# ── Loaders ───────────────────────────────────

def load_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(filepath)
    pages  = [page.extract_text() or "" for page in reader.pages]
    return clean_text(" ".join(pages))


def load_text(filepath: str) -> str:
    """Load plain .txt or .md file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def load_html(filepath: str) -> str:
    """Extract text from saved HTML file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        return clean_text(soup.get_text(separator=" "))


def fetch_url(url: str) -> str:
    """Fetch and extract text from a live URL (for public docs)."""
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return clean_text(soup.get_text(separator=" "))
    except Exception as e:
        print(f"  ⚠  Could not fetch {url}: {e}")
        return ""


def load_tickets_csv(filepath: str) -> list:
    """Load CSV of IT support tickets as chunks."""
    df = pd.read_csv(filepath)
    chunks = []
    for _, row in df.iterrows():
        text = " | ".join([
            f"Title: {row.get('title','')}" ,
            f"Description: {row.get('description','')}",
            f"Resolution: {row.get('resolution','')}",
            f"Category: {row.get('category','')}",
        ])
        text = clean_text(text)
        if len(text) > 50:
            chunks.append({
                "id":     f"ticket__{row.get('id', _)}",
                "source": f"Ticket #{row.get('id', _)}",
                "text":   text,
            })
    return chunks


# ── Main Ingestion ────────────────────────────

def ingest_all():
    """
    Full ingestion pipeline:
    1. Load all documents
    2. Chunk them
    3. Embed with sentence-transformers
    4. Store in FAISS index + metadata JSON
    """
    print("\n🚀 Starting ingestion pipeline...\n")
    os.makedirs("data", exist_ok=True)
    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)

    all_chunks = []

    # ── A. Load local documents ──
    docs_dir = Path(config.DOCS_DIR)
    if docs_dir.exists():
        files = list(docs_dir.rglob("*"))
        print(f"📁 Found {len(files)} files in {docs_dir}")
        for fp in tqdm(files, desc="Loading docs"):
            fp = str(fp)
            try:
                if fp.endswith(".pdf"):
                    text = load_pdf(fp)
                elif fp.endswith(".html") or fp.endswith(".htm"):
                    text = load_html(fp)
                elif fp.endswith((".txt", ".md")):
                    text = load_text(fp)
                else:
                    continue
                source = os.path.basename(fp)
                chunks = chunk_text(text, source)
                all_chunks.extend(chunks)
                print(f"  ✓ {source} → {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ Error loading {fp}: {e}")

    # ── B. Load tickets CSV ──
    tickets_path = Path(config.TICKETS_CSV)
    if tickets_path.exists():
        print(f"\n🎫 Loading tickets from {tickets_path}")
        ticket_chunks = load_tickets_csv(str(tickets_path))
        all_chunks.extend(ticket_chunks)
        print(f"  ✓ Loaded {len(ticket_chunks)} ticket records")

    # ── C. Fetch sample public docs (demo mode) ──
    if not all_chunks:
        print("\n⚠  No local docs found. Fetching sample public docs for demo...")
        sample_urls = {
            "docker-overview": "https://docs.docker.com/get-started/overview/",
            "kubernetes-concepts": "https://kubernetes.io/docs/concepts/overview/",
        }
        for name, url in sample_urls.items():
            print(f"  🌐 Fetching {name}...")
            text = fetch_url(url)
            if text:
                chunks = chunk_text(text, name)
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} chunks")
            time.sleep(1)

    sample_chunks = _get_sample_chunks()
    all_chunks.extend(sample_chunks)
    print(f"  ✓ Added {len(sample_chunks)} built-in sample chunks")

    print(f"\n📊 Total chunks to embed: {len(all_chunks)}")

    # ── D. Generate embeddings ──
    print(f"\n🔢 Loading embedding model: {config.EMBEDDING_MODEL}")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    print("⚙  Generating embeddings (this may take a minute)...")
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # ── E. Build FAISS index ──
    print("\n💾 Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Inner product (cosine after normalize)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    index_file = os.path.join(config.FAISS_INDEX_PATH, "index.bin")
    faiss.write_index(index, index_file)
    print(f"  ✓ FAISS index saved → {index_file} ({index.ntotal} vectors)")

    # ── F. Save metadata ──
    with open(config.METADATA_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"  ✓ Metadata saved → {config.METADATA_PATH}")

    print(f"\n✅ Ingestion complete! {len(all_chunks)} chunks indexed.\n")
    return len(all_chunks)


def _get_sample_chunks():
    """Built-in sample data for demo without internet."""
    samples = [
        ("docker-intro",
         "Docker is an open platform for developing, shipping, and running applications. "
         "Docker enables you to separate your applications from your infrastructure so you can "
         "deliver software quickly. With Docker, you can manage your infrastructure in the same "
         "ways you manage your applications. A Docker container image is a lightweight, standalone, "
         "executable package of software that includes everything needed to run an application: "
         "code, runtime, system tools, system libraries and settings."),

        ("kubernetes-pods",
         "A Pod is the smallest deployable unit of computing that you can create and manage in Kubernetes. "
         "A Pod is a group of one or more containers, with shared storage and network resources, and a "
         "specification for how to run the containers. To restart a crashed pod use: kubectl delete pod <pod-name> "
         "and Kubernetes will automatically recreate it. To check pod status use: kubectl get pods -n <namespace>."),

        ("kafka-consumer",
         "A Kafka consumer reads records from a Kafka topic. Consumers subscribe to topics and process "
         "the feed of published messages. To set up a Kafka consumer in Python: use the confluent-kafka library. "
         "Create a Consumer instance with bootstrap.servers pointing to your Kafka broker. Call subscribe() with "
         "a list of topic names. Then loop calling poll() to receive messages. Always commit offsets after processing."),

        ("ticket-001",
         "Title: Cannot connect to database | Description: Getting connection refused error on port 5432 | "
         "Resolution: Check if PostgreSQL service is running. Run: sudo systemctl start postgresql. "
         "Also verify firewall rules allow port 5432. Category: Database"),

        ("ticket-002",
         "Title: Docker container keeps restarting | Description: Container exits with code 137 | "
         "Resolution: Exit code 137 means OOM (Out of Memory). Increase Docker memory limit in "
         "docker-compose.yml under mem_limit. Or check for memory leaks in application. Category: Infrastructure"),

        ("onboarding-sop",
         "New Engineer Onboarding SOP: Step 1 - Request VPN access from IT helpdesk. "
         "Step 2 - Clone the main repository from internal GitLab. "
         "Step 3 - Install required tools: Docker, kubectl, Python 3.11. "
         "Step 4 - Run setup.sh to configure local development environment. "
         "Step 5 - Complete security training module on LMS portal. "
         "Step 6 - Schedule 1-on-1 with team lead within first week."),

        ("fastapi-setup",
         "FastAPI is a modern, fast web framework for building APIs with Python. "
         "To install FastAPI run: pip install fastapi uvicorn. "
         "Create a main.py with: from fastapi import FastAPI; app = FastAPI(). "
         "Define routes using decorators: @app.get('/') def root(): return {'message': 'Hello'}. "
         "Run the server with: uvicorn main:app --reload. "
         "FastAPI automatically generates OpenAPI docs at /docs endpoint."),

        ("python-venv",
         "Python virtual environments isolate project dependencies. "
         "To create a virtual environment: python -m venv venv. "
         "To activate on Linux/Mac: source venv/bin/activate. "
         "To activate on Windows: venv\\Scripts\\activate. "
         "Install packages: pip install -r requirements.txt. "
         "Deactivate with: deactivate command."),
    ]

    all_chunks = []
    for source, text in samples:
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    ingest_all()
