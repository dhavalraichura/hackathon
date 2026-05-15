# ─────────────────────────────────────────────
# check_dependencies.py
# Run this to verify your setup before starting the app.
# Usage: python check_dependencies.py
# ─────────────────────────────────────────────

import sys
import shutil
import os

SEP = "─" * 52


def check_pip(display_name: str, import_name: str, install_cmd: str = "") -> bool:
    try:
        __import__(import_name)
        print(f"  ✅  {display_name}")
        return True
    except ImportError:
        cmd = install_cmd or f"pip install {display_name.lower()}"
        print(f"  ❌  {display_name}  →  pip install {cmd}")
        return False


def check_bin(binary: str, install_hint: str = "") -> bool:
    if shutil.which(binary):
        print(f"  ✅  {binary}  (found on PATH)")
        return True
    else:
        hint = f"  hint: {install_hint}" if install_hint else ""
        print(f"  ❌  {binary}  not found on PATH.{hint}")
        return False


def check_win_path(name: str, path: str) -> bool:
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status}  {name}: {path}")
    return exists


# ════════════════════════════════════════════
print(f"\n{'═'*52}")
print(f"  Dependency Checker  —  {sys.platform}  /  Python {sys.version.split()[0]}")
print(f"{'═'*52}\n")

# ── Python packages ──────────────────────────
print("📦 Python packages")
print(SEP)
pip_checks = [
    ("sentence-transformers",  "sentence_transformers",  "sentence-transformers"),
    ("transformers",           "transformers",            "transformers"),
    ("torch",                  "torch",                   "torch"),
    ("faiss-cpu",              "faiss",                   "faiss-cpu"),
    ("pypdf",                  "pypdf",                   "pypdf"),
    ("pdfplumber",             "pdfplumber",              "pdfplumber"),
    ("pdf2image",              "pdf2image",               "pdf2image"),
    ("pytesseract",            "pytesseract",             "pytesseract"),
    ("Pillow",                 "PIL",                     "Pillow"),
    ("python-docx",            "docx",                    "python-docx"),
    ("beautifulsoup4",         "bs4",                     "beautifulsoup4"),
    ("pandas",                 "pandas",                  "pandas"),
    ("numpy",                  "numpy",                   "numpy"),
    ("openai-whisper",         "whisper",                 "openai-whisper"),
    ("moviepy",                "moviepy",                 "moviepy"),
    ("imageio-ffmpeg",         "imageio_ffmpeg",          "imageio-ffmpeg"),
    ("streamlit",              "streamlit",               "streamlit"),
    ("scikit-learn",           "sklearn",                 "scikit-learn"),
    ("python-dotenv",          "dotenv",                  "python-dotenv"),
    ("requests",               "requests",                "requests"),
    ("tqdm",                   "tqdm",                    "tqdm"),
    ("tabulate",               "tabulate",                "tabulate"),
    ("ollama",                 "ollama",                  "ollama"),
    ("openai",                 "openai",                  "openai"),
    ("anthropic",              "anthropic",               "anthropic"),
]
pip_results = [check_pip(d, i, c) for d, i, c in pip_checks]
print()

# ── System binaries ──────────────────────────
print("🔧 System binaries")
print(SEP)

if sys.platform == "win32":
    # On Windows, tesseract/poppler/ffmpeg may not be on PATH —
    # check the known install paths instead
    from media_ingest import TESSERACT_PATH_WIN, POPPLER_PATH_WIN
    check_win_path("Tesseract",            TESSERACT_PATH_WIN)
    check_win_path("Poppler (bin folder)", POPPLER_PATH_WIN)
    # ffmpeg is handled by imageio-ffmpeg on Windows
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"  ✅  ffmpeg  (via imageio-ffmpeg: {ffmpeg_exe})")
    except Exception:
        print("  ❌  ffmpeg  →  pip install imageio-ffmpeg")
else:
    check_bin("tesseract",
              "Linux: sudo apt-get install tesseract-ocr  |  Mac: brew install tesseract")
    check_bin("pdftoppm",   # part of poppler-utils, used by pdf2image
              "Linux: sudo apt-get install poppler-utils  |  Mac: brew install poppler")
    check_bin("ffmpeg",
              "Linux: sudo apt-get install ffmpeg         |  Mac: brew install ffmpeg")

print()

# ── Ollama connectivity ───────────────────────
print("🤖 Ollama")
print(SEP)
try:
    import requests
    resp = requests.get("http://localhost:11434", timeout=3)
    print("  ✅  Ollama server reachable at http://localhost:11434")
    # List available models
    models_resp = requests.get("http://localhost:11434/api/tags", timeout=3)
    if models_resp.ok:
        models = [m["name"] for m in models_resp.json().get("models", [])]
        if models:
            print(f"  📋  Available models: {', '.join(models)}")
        else:
            print("  ⚠   No models pulled yet. Run: ollama pull gemma4")
except Exception:
    print("  ❌  Ollama not running. Start with: ollama serve")

print()

# ── Summary ──────────────────────────────────
print("📊 Summary")
print(SEP)
n_ok   = sum(pip_results)
n_fail = len(pip_results) - n_ok
print(f"  Python packages:  {n_ok}/{len(pip_results)} installed")

if n_fail == 0:
    print("  🎉  All good! Run: streamlit run app.py")
else:
    print(f"  ⚠   {n_fail} package(s) missing. Install them and re-run this check.")

if sys.platform == "win32":
    print()
    print("  Windows install guides:")
    print("  • Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  • Poppler:   https://github.com/oschwartz10612/poppler-windows/releases")
    print("  • After installing, update TESSERACT_PATH_WIN / POPPLER_PATH_WIN in media_ingest.py")

print()