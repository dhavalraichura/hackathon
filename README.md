# 🧠 Enterprise Knowledge Assistant
### NASSCOM Hackathon — Use Case 2: RAG + Agentic Workflow

A fully open-source Enterprise Knowledge Copilot that answers employee queries
by intelligently searching internal documents, SOPs, and support tickets using
RAG (Retrieval-Augmented Generation) with a ReAct agentic workflow.

---

## 📁 Project Structure

```
enterprise_knowledge_assistant/
│
├── app.py                    # Streamlit Web UI (main interface)
├── agent.py                  # ReAct agentic engine (4 tools)
├── retriever.py              # FAISS vector search + reranking
├── llm.py                    # LLM interface (Ollama/OpenAI/Anthropic)
├── ingest.py                 # Data ingestion pipeline
├── evaluate.py               # Precision, Recall, F1, LLM-as-Judge
├── generate_sample_data.py   # Generate synthetic tickets + SOPs
├── config.py                 # Central configuration
├── requirements.txt          # All dependencies
├── .env.example              # Environment variables template
└── data/
    ├── documents/            # Put your PDFs/TXTs here
    ├── tickets.csv           # IT support tickets
    ├── faiss_index/          # Auto-generated FAISS index
    ├── metadata.json         # Auto-generated chunk metadata
    └── eval_dataset.json     # Evaluation test set
```

---

## ⚡ Quick Start (5 Steps)

### Step 1 — Clone & Install

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Setup LLM (Choose ONE option)

#### Option A — Ollama (FREE, 100% local, recommended)
```bash
# Install Ollama from https://ollama.com
# Then pull LLaMA 3:
ollama pull llama3

# Start Ollama server (keep this running in background)
ollama serve
```

#### Option B — OpenAI API
```bash
# Create .env file
echo "LLM_PROVIDER=openai" > .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

#### Option C — Anthropic Claude API
```bash
echo "LLM_PROVIDER=anthropic" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env
```

### Step 3 — Generate Sample Data (if you don't have real docs)

```bash
python generate_sample_data.py
```

This creates:
- `data/tickets.csv` — 30 synthetic IT support tickets
- `data/documents/` — 3 SOP text files (onboarding, deployment, incidents)
- `data/eval_dataset.json` — 8 evaluation questions

**OR add your own documents:**
```bash
# Copy your PDFs/TXTs into the documents folder
cp your-docs/*.pdf data/documents/
cp your-docs/*.txt data/documents/
```

### Step 4 — Build the Knowledge Index

```bash
python ingest.py
```

This will:
- Load all documents from `data/documents/`
- Load tickets from `data/tickets.csv`
- Split into chunks of ~400 words
- Generate embeddings using `all-MiniLM-L6-v2`
- Build and save FAISS index

Expected output:
```
🚀 Starting ingestion pipeline...
📁 Found 3 files in data/documents
✓ onboarding_sop.txt → 8 chunks
✓ deployment_sop.txt → 6 chunks
✓ incident_response_sop.txt → 7 chunks
🎫 Loading tickets from data/tickets.csv
✓ Loaded 30 ticket records
📊 Total chunks to embed: 51
⚙  Generating embeddings...
💾 Building FAISS index...
✅ Ingestion complete! 51 chunks indexed.
```

### Step 5 — Run the App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🧪 Running Evaluation

```bash
# Basic evaluation (Precision, Recall, F1)
python evaluate.py

# With semantic similarity (calls LLM — takes longer)
python evaluate.py --llm-judge
```

Expected output:
```
╭─────────────────────────────────────────────╮
│  📊 ENTERPRISE KNOWLEDGE ASSISTANT          │
│        EVALUATION REPORT                    │
╰─────────────────────────────────────────────╯

📐 RETRIEVAL METRICS
┌─────────────┬───────┬───────────────┐
│ Metric      │ Score │ Target        │
├─────────────┼───────┼───────────────┤
│ Precision@k │ 0.800 │ ≥ 0.70 target │
│ Recall@k    │ 0.750 │ ≥ 0.65 target │
│ F1 Score    │ 0.774 │ ≥ 0.75 target │
└─────────────┴───────┴───────────────┘

🏁 VERDICT: ✅ EXCELLENT — Ready for demo
```

---

## 🔬 Architecture Overview

```
User Query
    ↓
[Streamlit UI / FastAPI]
    ↓
[ReAct Agent]
  THINK → ACT → OBSERVE → loop
    ↓            ↓
  [Tool 1]   [Tool 2]   [Tool 3]   [Tool 4]
  Doc Search  Tickets   Summarizer  Escalate
    ↓
[Retriever]
  - sentence-transformers embedding
  - FAISS top-k search
  - CrossEncoder reranking (optional)
    ↓
[LLM] (Ollama/OpenAI/Anthropic)
  - RAG prompt with context
  - Source citations
  - Hallucination guardrail
    ↓
Answer + Sources + ReAct Steps
```

---

## 🛠️ Open Source Tools Used

| Tool | Version | Purpose | License |
|------|---------|---------|---------|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | 2.7.0 | Text embeddings (`all-MiniLM-L6-v2`) | Apache 2.0 |
| [FAISS](https://github.com/facebookresearch/faiss) | 1.8.0 | Vector similarity search | MIT |
| [LangChain](https://github.com/langchain-ai/langchain) | 0.2.1 | Agent orchestration framework | MIT |
| [Ollama](https://ollama.com) | latest | Run LLaMA3 locally (free, on-premise) | MIT |
| [LLaMA 3](https://llama.meta.com) | 8B | Open-source LLM by Meta | Meta Llama 3 License |
| [Streamlit](https://streamlit.io) | 1.35.0 | Web UI | Apache 2.0 |
| [pypdf](https://github.com/py-pdf/pypdf) | 4.2.0 | PDF text extraction | BSD |
| [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | 4.12.3 | HTML parsing | MIT |
| [FastAPI](https://fastapi.tiangolo.com) | 0.111.0 | REST API | MIT |
| [scikit-learn](https://scikit-learn.org) | 1.4.2 | Precision/Recall/F1 metrics | BSD |
| [HuggingFace Transformers](https://huggingface.co/transformers) | 4.40.0 | Model loading | Apache 2.0 |
| [CrossEncoder/ms-marco](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | - | Reranking (bonus) | Apache 2.0 |
| [OpenAI Whisper](https://github.com/openai/whisper) | 20231117 | On-premise video transcription | MIT |
| [pandas](https://pandas.pydata.org) | 2.2.2 | CSV ticket processing | BSD |
| [PyTorch](https://pytorch.org) | 2.2.2 | ML backend | BSD |

### Documentation / Data Sources
| Source | URL | Used For |
|--------|-----|---------|
| Apache Kafka Docs | https://kafka.apache.org/documentation/ | Sample knowledge base |
| Kubernetes Docs | https://kubernetes.io/docs/ | Sample knowledge base |
| Docker Docs | https://docs.docker.com/ | Sample knowledge base |
| FastAPI Docs | https://fastapi.tiangolo.com/ | Sample knowledge base |
| SQuAD Dataset | https://rajpurkar.github.io/SQuAD-explorer/ | QA evaluation benchmark |
| StackOverflow (Kaggle) | https://www.kaggle.com/datasets/stackoverflow/stackoverflow | Ticket simulation reference |

---

## ⚙️ Configuration (.env file)

```env
# LLM Provider: ollama | openai | anthropic
LLM_PROVIDER=ollama
LLM_MODEL=llama3

# API Keys (only needed if not using Ollama)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Ollama server URL
OLLAMA_URL=http://localhost:11434
```

---

## 🔒 Data Security

- **100% on-premise** when using Ollama — no data leaves your network
- Whisper transcription runs locally — audio never sent to external servers
- FAISS index stored locally in `data/faiss_index/`
- Compliant with: GDPR, India DPDP Act, HIPAA (when self-hosted)

---

## 📊 Evaluation Metrics Explained

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Precision@k | relevant_retrieved / k | Quality of retrieved chunks |
| Recall@k | relevant_retrieved / total_relevant | Coverage of relevant docs |
| F1 Score | 2 × P × R / (P + R) | Balance of precision & recall |
| Semantic Similarity | cosine(answer_vec, expected_vec) | Answer quality |
| LLM-as-Judge | GPT/Claude grades answer 1-5 | End-to-end quality |

---

## 🚀 Adding Your Own Documents

```bash
# PDFs
cp your-sop.pdf data/documents/

# Text files
cp your-wiki.txt data/documents/

# Re-index after adding docs
python ingest.py
```

---

## 🎬 Adding Video Transcription (Optional)

```bash
# Install Whisper
pip install openai-whisper

# Transcribe a video (runs 100% locally)
python -c "
import whisper
model = whisper.load_model('base')
result = model.transcribe('training_video.mp4')
with open('data/documents/training_video.txt', 'w') as f:
    f.write(result['text'])
print('Done! Now run: python ingest.py')
"
```

---

## 📝 License

MIT License — Open source, free to use and modify.

---

*Built for NASSCOM Hackathon 2025 | Use Case 2: Enterprise Knowledge Assistant*
