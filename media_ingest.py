# ─────────────────────────────────────────────
# media_ingest.py — Multi-modal ingestion pipeline
# Handles: Videos → Transcript → Embeddings
#          Images → OCR Text → Embeddings
#          PDFs   → OCR + Text → Embeddings
#          DOCX / TXT / MD / CSV → Embeddings
#
# Cross-platform: Windows, Linux, Mac
# ─────────────────────────────────────────────

import os
import sys
import json
import re
import tempfile
import numpy as np
import faiss

from pathlib import Path
from typing import Optional

import config

# ══════════════════════════════════════════════
# WINDOWS BINARY CONFIGURATION
# Edit these paths if you installed to non-default locations
# ══════════════════════════════════════════════

TESSERACT_PATH_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH_WIN   = r"C:\poppler\Library\bin"

# Auto-configure Tesseract path on Windows
if sys.platform == "win32":
    try:
        import pytesseract
        if os.path.exists(TESSERACT_PATH_WIN):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_WIN
    except ImportError:
        pass

    # Use imageio-ffmpeg as bundled ffmpeg for moviepy on Windows
    try:
        import imageio_ffmpeg
        os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass


def _get_poppler_path() -> Optional[str]:
    """Return poppler bin path on Windows; None on Linux/Mac (uses system PATH)."""
    if sys.platform == "win32":
        if os.path.exists(POPPLER_PATH_WIN):
            return POPPLER_PATH_WIN
        print("  ⚠  Poppler not found at expected Windows path. "
              "Download from: https://github.com/oschwartz10612/poppler-windows/releases")
        return None
    return None  # Linux/Mac find it via PATH automatically


# ── Optional heavy imports (graceful fallback) ─────────────

def _try_import(module: str):
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        return None


# ══════════════════════════════════════════════
# TEXT CLEANING & CHUNKING
# ══════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def chunk_text(text: str, source: str,
               chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    words    = text.split()
    chunks   = []
    i        = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_str   = " ".join(chunk_words)
        if len(chunk_words) > 30:
            chunks.append({
                "id":     f"{source}__chunk_{chunk_id}",
                "source": source,
                "text":   chunk_str,
            })
            chunk_id += 1
        i += chunk_size - overlap
    return chunks


# ══════════════════════════════════════════════
# VIDEO → TRANSCRIPT
# ══════════════════════════════════════════════

def transcribe_video(filepath: str, source_name: str) -> list[dict]:
    """
    Convert a video file to text chunks via Whisper.
    Extracts audio with moviepy, transcribes with openai-whisper.

    Install:
        pip install openai-whisper moviepy imageio-ffmpeg
        Linux/Mac only: brew install ffmpeg  OR  apt install ffmpeg
        Windows: imageio-ffmpeg handles ffmpeg automatically
    """
    whisper = _try_import("whisper")
    if whisper is None:
        return _error_chunk(source_name,
            "openai-whisper not installed. Run: pip install openai-whisper")

    moviepy_editor = _try_import("moviepy.editor")
    if moviepy_editor is None:
        return _error_chunk(source_name,
            "moviepy not installed. Run: pip install moviepy imageio-ffmpeg")

    print(f"  🎬 Extracting audio from {source_name}...")
    audio_path = None
    try:
        video = moviepy_editor.VideoFileClip(filepath)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
    except Exception as e:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        return _error_chunk(source_name, f"Audio extraction failed: {e}")

    print(f"  🗣  Transcribing with Whisper (base model)...")
    try:
        # Use "small" or "medium" for better accuracy on noisy/accented audio
        model      = whisper.load_model("base")
        result     = model.transcribe(audio_path)
        transcript = result["text"]
    except Exception as e:
        return _error_chunk(source_name, f"Whisper transcription failed: {e}")
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    if not transcript.strip():
        return _error_chunk(source_name, "Whisper returned an empty transcript")

    print(f"  ✓ Transcript: {len(transcript.split())} words")
    text   = clean_text(transcript)
    chunks = chunk_text(text, source_name)
    for c in chunks:
        c["media_type"] = "video_transcript"
    return chunks


# ══════════════════════════════════════════════
# IMAGE → OCR TEXT
# ══════════════════════════════════════════════

def extract_image_text(filepath: str, source_name: str) -> list[dict]:
    """
    OCR an image file using pytesseract + Pillow.

    Install:
        pip install pytesseract Pillow
        Linux:   sudo apt-get install tesseract-ocr
        Mac:     brew install tesseract
        Windows: https://github.com/UB-Mannheim/tesseract/wiki
                 Then set TESSERACT_PATH_WIN at top of this file.
    """
    pytesseract = _try_import("pytesseract")
    pil_image   = _try_import("PIL.Image")

    if pytesseract is None or pil_image is None:
        return _error_chunk(source_name,
            "pytesseract/Pillow not installed. Run: pip install pytesseract Pillow")

    print(f"  🔍 OCR scanning image: {source_name}...")
    try:
        image = pil_image.open(filepath)
        # Convert to RGB to handle PNG transparency / TIFF multi-frame
        image = image.convert("RGB")
        text  = pytesseract.image_to_string(image)
    except Exception as e:
        return _error_chunk(source_name, f"Image OCR failed: {e}")

    text = clean_text(text)
    if len(text.split()) < 10:
        return _error_chunk(source_name,
            "OCR found very little text in image — check image quality or rotation")

    print(f"  ✓ Extracted {len(text.split())} words from image")
    chunks = chunk_text(text, source_name)
    for c in chunks:
        c["media_type"] = "image_ocr"
    return chunks


# ══════════════════════════════════════════════
# PDF → TEXT LAYER + OCR FALLBACK
# ══════════════════════════════════════════════

def extract_pdf_text(filepath: str, source_name: str,
                     ocr_fallback: bool = True) -> list[dict]:
    """
    Extract text from PDF:
      1. Try pypdf (fast — works on text-based / digital PDFs)
      2. If text is sparse (< 50 words/page avg), fall back to OCR

    Install (base):  pip install pypdf pdfplumber
    Install (OCR):   pip install pdf2image pytesseract Pillow
                     Linux:   sudo apt-get install tesseract-ocr poppler-utils
                     Mac:     brew install tesseract poppler
                     Windows: install Tesseract + Poppler manually (see README)
    """
    from pypdf import PdfReader

    print(f"  📄 Extracting PDF: {source_name}...")
    try:
        reader     = PdfReader(filepath)
        pages_text = [p.extract_text() or "" for p in reader.pages]
        full_text  = " ".join(pages_text)
        n_pages    = max(len(reader.pages), 1)
        avg_words  = len(full_text.split()) / n_pages
    except Exception as e:
        return _error_chunk(source_name, f"pypdf failed: {e}")

    # Scanned PDFs have very little extractable text — trigger OCR
    if avg_words < 50 and ocr_fallback:
        print(f"  ⚠  Low text density ({avg_words:.0f} words/page) — switching to OCR...")
        return _ocr_pdf(filepath, source_name)

    text = clean_text(full_text)
    print(f"  ✓ Extracted {len(text.split())} words from PDF (text layer)")
    chunks = chunk_text(text, source_name)
    for c in chunks:
        c["media_type"] = "pdf_text"
    return chunks


def _ocr_pdf(filepath: str, source_name: str) -> list[dict]:
    """OCR a scanned PDF by rasterising each page then running Tesseract."""
    pdf2image   = _try_import("pdf2image")
    pytesseract = _try_import("pytesseract")

    if pdf2image is None or pytesseract is None:
        return _error_chunk(source_name,
            "OCR libs missing. Run: pip install pdf2image pytesseract Pillow\n"
            "Linux: sudo apt-get install tesseract-ocr poppler-utils\n"
            "Mac:   brew install tesseract poppler\n"
            "Win:   see README for Tesseract + Poppler installers")

    print(f"  🔍 OCR scanning {source_name} page by page...")
    try:
        images = pdf2image.convert_from_path(
            filepath,
            dpi=300,
            poppler_path=_get_poppler_path(),   # None on Linux/Mac = use PATH
        )
        pages_text = []
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            pages_text.append(f"[Page {i + 1}]\n{page_text}")
            print(f"     OCR page {i + 1}/{len(images)}", end="\r")
        print()
    except Exception as e:
        return _error_chunk(source_name, f"PDF OCR failed: {e}")

    full_text = clean_text(" ".join(pages_text))
    print(f"  ✓ OCR extracted {len(full_text.split())} words from scanned PDF")
    chunks = chunk_text(full_text, source_name)
    for c in chunks:
        c["media_type"] = "pdf_ocr"
    return chunks


# ══════════════════════════════════════════════
# DOCX
# ══════════════════════════════════════════════

def extract_docx_text(filepath: str, source_name: str) -> list[dict]:
    """
    Extract text from Word .docx files including tables.
    Install: pip install python-docx
    """
    docx = _try_import("docx")
    if docx is None:
        return _error_chunk(source_name,
            "python-docx not installed. Run: pip install python-docx")

    print(f"  📝 Extracting DOCX: {source_name}...")
    try:
        doc = docx.Document(filepath)

        # Extract paragraphs
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Also extract table cells
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    paras.append(row_text)

        text = clean_text(" ".join(paras))
    except Exception as e:
        return _error_chunk(source_name, f"DOCX extraction failed: {e}")

    if len(text.split()) < 10:
        return _error_chunk(source_name, "DOCX appears to be empty or unreadable")

    print(f"  ✓ Extracted {len(text.split())} words from DOCX")
    chunks = chunk_text(text, source_name)
    for c in chunks:
        c["media_type"] = "docx"
    return chunks


# ══════════════════════════════════════════════
# PLAIN TEXT / HTML / MARKDOWN
# ══════════════════════════════════════════════

def extract_text_file(filepath: str, source_name: str, ext: str) -> list[dict]:
    """Extract text from .txt / .md / .html files."""
    try:
        if ext in {".html", ".htm"}:
            bs4 = _try_import("bs4")
            if bs4 is None:
                return _error_chunk(source_name,
                    "beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                soup = bs4.BeautifulSoup(f.read(), "html.parser")
                text = clean_text(soup.get_text(separator=" "))
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = clean_text(f.read())
    except Exception as e:
        return _error_chunk(source_name, f"Text file read failed: {e}")

    print(f"  ✓ Extracted {len(text.split())} words from {ext} file")
    chunks = chunk_text(text, source_name)
    for c in chunks:
        c["media_type"] = "text"
    return chunks


# ══════════════════════════════════════════════
# CSV / TICKETS
# ══════════════════════════════════════════════

def extract_csv(filepath: str, source_name: str) -> list[dict]:
    """Load CSV rows as text chunks (works for tickets or any tabular data)."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
    except Exception as e:
        return _error_chunk(source_name, f"CSV read failed: {e}")

    chunks = []
    for _, row in df.iterrows():
        text = " | ".join(
            f"{col}: {val}"
            for col, val in row.items()
            if str(val).strip() and str(val).strip().lower() != "nan"
        )
        text = clean_text(text)
        if len(text.split()) > 5:
            row_id = row.get("id", len(chunks))
            chunks.append({
                "id":         f"{source_name}__row_{row_id}",
                "source":     source_name,
                "text":       text,
                "media_type": "csv",
            })

    print(f"  ✓ Extracted {len(chunks)} rows from CSV")
    return chunks


# ══════════════════════════════════════════════
# DISPATCHER — single entry point for all files
# ══════════════════════════════════════════════

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS   = {".pdf"}
DOCX_EXTS  = {".docx"}
TEXT_EXTS  = {".txt", ".md", ".html", ".htm"}
CSV_EXTS   = {".csv"}

ALL_SUPPORTED = VIDEO_EXTS | IMAGE_EXTS | PDF_EXTS | DOCX_EXTS | TEXT_EXTS | CSV_EXTS


def process_uploaded_file(filepath: str,
                           source_name: Optional[str] = None) -> list[dict]:
    """
    Auto-detect file type and extract text chunks.
    Single entry point for all uploaded files.
    """
    ext         = Path(filepath).suffix.lower()
    source_name = source_name or Path(filepath).name

    if ext in VIDEO_EXTS:
        return transcribe_video(filepath, source_name)
    elif ext in IMAGE_EXTS:
        return extract_image_text(filepath, source_name)
    elif ext in PDF_EXTS:
        return extract_pdf_text(filepath, source_name, ocr_fallback=True)
    elif ext in DOCX_EXTS:
        return extract_docx_text(filepath, source_name)
    elif ext in TEXT_EXTS:
        return extract_text_file(filepath, source_name, ext)
    elif ext in CSV_EXTS:
        return extract_csv(filepath, source_name)
    else:
        return _error_chunk(source_name,
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED))}")


# ══════════════════════════════════════════════
# LIVE INDEX UPDATER
# Appends new chunks to existing FAISS index + metadata
# ══════════════════════════════════════════════

def add_chunks_to_index(new_chunks: list[dict]) -> int:
    """
    Embed new chunks and append them to the existing FAISS index + metadata JSON.
    Deduplicates by chunk id so re-uploading the same file is safe.
    Returns number of new vectors added.
    """
    if not new_chunks:
        return 0

    from sentence_transformers import SentenceTransformer

    print(f"  🔢 Embedding {len(new_chunks)} chunks...")
    embedder   = SentenceTransformer(config.EMBEDDING_MODEL)
    texts      = [c["text"] for c in new_chunks]
    embeddings = embedder.encode(texts, batch_size=16, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    # Load existing index + metadata (or create fresh)
    index_file = os.path.join(config.FAISS_INDEX_PATH, "index.bin")
    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)

    if os.path.exists(index_file) and os.path.exists(config.METADATA_PATH):
        index = faiss.read_index(index_file)
        with open(config.METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        dim      = embeddings.shape[1]
        index    = faiss.IndexFlatIP(dim)
        metadata = []

    # Skip chunks already indexed (safe re-upload)
    existing_ids = {c["id"] for c in metadata}
    fresh_pairs  = [
        (c, e) for c, e in zip(new_chunks, embeddings)
        if c["id"] not in existing_ids
    ]

    if not fresh_pairs:
        print("  ℹ  All chunks already indexed — nothing new added.")
        return 0

    fresh_chunks, fresh_vecs = zip(*fresh_pairs)
    fresh_vecs = np.array(fresh_vecs).astype("float32")

    index.add(fresh_vecs)
    metadata.extend(fresh_chunks)

    faiss.write_index(index, index_file)
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Added {len(fresh_chunks)} vectors. Index total: {index.ntotal}")
    return len(fresh_chunks)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def _error_chunk(source: str, message: str) -> list[dict]:
    """Return a single descriptive error chunk so callers always get a list."""
    print(f"  ✗ [{source}] {message}")
    return [{
        "id":         f"{source}__error",
        "source":     source,
        "text":       f"[Processing error for {source}: {message}]",
        "media_type": "error",
    }]


def get_supported_extensions() -> str:
    return ", ".join(sorted(ALL_SUPPORTED))