# ─────────────────────────────────────────────
# upload_panel.py — Streamlit file upload widget
#
# Usage in app.py:
#   from upload_panel import render_upload_panel
#   # inside `with st.sidebar:` block:
#   render_upload_panel()
# ─────────────────────────────────────────────

import os
import tempfile
import streamlit as st
from pathlib import Path

from media_ingest import (
    process_uploaded_file,
    add_chunks_to_index,
    VIDEO_EXTS,
    IMAGE_EXTS,
    PDF_EXTS,
    DOCX_EXTS,
    TEXT_EXTS,
    CSV_EXTS,
)

# Human-readable labels for each media type tag
MEDIA_TYPE_LABELS = {
    "video_transcript": "🎬 Video transcript",
    "image_ocr":        "🖼  Image (OCR)",
    "pdf_text":         "📄 PDF (text layer)",
    "pdf_ocr":          "📄 PDF (scanned OCR)",
    "docx":             "📝 Word document",
    "text":             "📃 Text / Markdown",
    "csv":              "📊 CSV / Tickets",
    "error":            "❌ Error",
}


def render_upload_panel():
    """
    Render the upload panel inside the Streamlit sidebar.
    Call this inside a `with st.sidebar:` block in app.py.
    """
    st.divider()
    st.subheader("📤 Upload Documents")
    st.caption("PDF, DOCX, TXT, MD, CSV, PNG, JPG, MP4, MOV and more")

    uploaded_files = st.file_uploader(
        label="Drop files here",
        accept_multiple_files=True,
        type=[
            # Videos
            "mp4", "mov", "avi", "mkv", "webm", "m4v", "flv",
            # Images
            "png", "jpg", "jpeg", "bmp", "tiff", "webp",
            # Documents
            "pdf", "docx", "txt", "md", "html", "csv",
        ],
        help="Files are processed locally — nothing is sent to the cloud.",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        return

    if st.button("⚙️ Process & Index Files",
                 use_container_width=True,
                 type="primary"):
        _process_files(uploaded_files)


def _process_files(uploaded_files):
    """Save each upload to a temp file, process it, and add to the FAISS index."""
    results_summary = []
    total_added     = 0
    n               = len(uploaded_files)

    progress = st.progress(0, text="Starting...")

    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        ext       = Path(file_name).suffix.lower()

        progress.progress(i / n, text=f"Processing {file_name}…")

        # Write buffer to a real temp file (required by most processing libs)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner(f"Extracting: {file_name}"):
                chunks = process_uploaded_file(tmp_path, source_name=file_name)

            if not chunks:
                results_summary.append((file_name, 0, "No content extracted"))
                continue

            error_chunks = [c for c in chunks if c.get("media_type") == "error"]
            good_chunks  = [c for c in chunks if c.get("media_type") != "error"]

            if error_chunks and not good_chunks:
                # Strip the "[Processing error: ...]" wrapper for display
                msg = error_chunks[0]["text"]
                results_summary.append((file_name, 0, msg))
                continue

            with st.spinner(f"Indexing: {file_name}"):
                added = add_chunks_to_index(good_chunks)

            media_type = good_chunks[0].get("media_type", "unknown") if good_chunks else "unknown"
            label      = MEDIA_TYPE_LABELS.get(media_type, media_type)
            results_summary.append(
                (file_name, added, f"{label} — {added} chunks indexed")
            )
            total_added += added

        except Exception as e:
            results_summary.append((file_name, 0, f"Unexpected error: {e}"))

        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    progress.progress(1.0, text="Done!")

    # ── Results summary ──
    st.markdown("---")
    st.markdown("**Results:**")
    for fname, count, msg in results_summary:
        if count > 0:
            st.success(f"**{fname}** — {msg}")
        else:
            st.error(f"**{fname}** — {msg}")

    if total_added > 0:
        st.info(
            f"✅ {total_added} new chunks added to the knowledge base. "
            "The assistant will use these files when answering questions."
        )
        # Reload the cached agent so retriever picks up the updated index
        st.cache_resource.clear()
        st.rerun()
    else:
        st.warning("No new content was indexed. See errors above.")