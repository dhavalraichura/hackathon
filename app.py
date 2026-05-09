# ─────────────────────────────────────────────
# app.py — Streamlit Web UI
# Enterprise Knowledge Assistant Chat Interface
# Run: streamlit run app.py
# ─────────────────────────────────────────────

import streamlit as st
import time
import os

import config

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
  .source-card {
    background: #1e2535;
    border-left: 3px solid #00d4ff;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85rem;
  }
  .step-card {
    background: #1a1f2e;
    border-left: 3px solid #7c3aed;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 3px 0;
    font-size: 0.8rem;
    font-family: monospace;
  }
  .escalate-box {
    background: #2d1515;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
  }
  .metric-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── Load Agent (cached) ───────────────────────
@st.cache_resource(show_spinner="🔧 Loading Knowledge Assistant...")
def load_agent(use_reranker):
    # Check if index exists, if not run ingestion
    index_file = os.path.join(config.FAISS_INDEX_PATH, "index.bin")
    if not os.path.exists(index_file):
        st.info("📥 First run: building knowledge index...")
        from ingest import ingest_all
        ingest_all()

    from agent import EnterpriseAgent
    return EnterpriseAgent(use_reranker=use_reranker)


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption("NASSCOM Hackathon — Use Case 2")
    st.divider()

    st.subheader("⚙️ Settings")
    use_reranker = st.toggle("Enable Reranker (bonus)", value=False,
                              help="Cross-encoder reranking improves precision. Slower.")
    show_steps   = st.toggle("Show ReAct Steps", value=True,
                              help="Show agent's Think → Act → Observe loop")
    show_sources = st.toggle("Show Sources", value=True)

    st.divider()
    st.subheader("📊 Evaluation")
    if st.button("▶ Run Evaluation Suite", use_container_width=True):
        st.session_state["run_eval"] = True

    st.divider()
    st.subheader("📁 Knowledge Base")
    st.caption(f"Embedding: `{config.EMBEDDING_MODEL}`")
    st.caption(f"Vector DB: FAISS (local)")
    st.caption(f"LLM: `{config.LLM_PROVIDER}/{config.LLM_MODEL}`")

    st.divider()
    if st.button("🔄 Re-index Documents", use_container_width=True):
        st.cache_resource.clear()
        with st.spinner("Re-indexing..."):
            from ingest import ingest_all
            count = ingest_all()
        st.success(f"✓ Indexed {count} chunks")

# ── Load Agent ────────────────────────────────
try:
    agent = load_agent(use_reranker)
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.info("Make sure you ran: `python ingest.py` first")
    st.stop()

# ── Evaluation Mode ───────────────────────────
if st.session_state.get("run_eval"):
    st.session_state["run_eval"] = False
    st.subheader("📊 Evaluation Report")
    with st.spinner("Running evaluation (this takes ~30 seconds)..."):
        from evaluate import Evaluator
        ev      = Evaluator()
        metrics = ev.run_full_evaluation(include_llm_judge=False)

    ret = metrics["retrieval"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision@k", f"{ret['precision']:.2%}", delta="target: 70%")
    with col2:
        st.metric("Recall@k", f"{ret['recall']:.2%}", delta="target: 65%")
    with col3:
        st.metric("F1 Score", f"{ret['f1']:.2%}", delta="target: 75%")

    st.subheader("Per-Question Breakdown")
    st.dataframe(
        ret["per_question"],
        use_container_width=True,
        column_config={
            "question":  st.column_config.TextColumn("Question", width="large"),
            "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1),
            "recall":    st.column_config.ProgressColumn("Recall",    min_value=0, max_value=1),
            "f1":        st.column_config.ProgressColumn("F1 Score",  min_value=0, max_value=1),
        }
    )
    st.divider()

# ── Chat Interface ────────────────────────────
st.header("💬 Ask the Knowledge Assistant")

# Sample questions
st.caption("Try asking:")
sample_qs = [
    "How do I restart a Kubernetes pod?",
    "How to set up a Kafka consumer in Python?",
    "My Docker container exits with code 137, how to fix?",
    "Steps for onboarding a new engineer",
    "How to create a Python virtual environment?",
]
cols = st.columns(len(sample_qs))
for col, q in zip(cols, sample_qs):
    if col.button(q[:30] + "…", use_container_width=True):
        st.session_state["prefill"] = q

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    score_pct = int(s.get('score', 0) * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{s["source"]}</b> — relevance: {score_pct}%<br>'
                        f'<small>{s["text"][:200]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        if msg.get("steps") and show_steps:
            with st.expander("🔄 ReAct Steps"):
                for step in msg["steps"]:
                    icon = {"think": "💭", "act": "⚡", "observe": "👁"}.get(step["step_type"], "•")
                    label = step.get("tool", step["step_type"].upper())
                    content = step.get("thought") or step.get("output", "")
                    st.markdown(
                        f'<div class="step-card">{icon} <b>{label}</b>: {content[:200]}</div>',
                        unsafe_allow_html=True
                    )

# Input
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask anything about your IT systems, SOPs, or past tickets...")

if prefill:
    user_input = prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            start_time = time.time()
            result     = agent.run(user_input)
            elapsed    = time.time() - start_time

        # Show answer
        if result.escalated:
            st.markdown(
                f'<div class="escalate-box">⚠️ <b>Escalated to Human Review</b></div>',
                unsafe_allow_html=True
            )
        st.markdown(result.answer)
        st.caption(f"⏱ {elapsed:.1f}s | {'✅ Confident' if result.confident else '⚠️ Low confidence'} | {len(result.sources)} sources")

        # Sources
        if result.sources and show_sources:
            with st.expander("📚 Sources"):
                for s in result.sources:
                    score_pct = int(s.get('score', 0) * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{s["source"]}</b> — relevance: {score_pct}%<br>'
                        f'<small>{s["text"][:200]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ReAct steps
        if result.steps and show_steps:
            with st.expander("🔄 ReAct Steps"):
                for step in result.steps:
                    icon = {"think": "💭", "act": "⚡"}.get(step.step_type, "•")
                    label = step.tool if step.tool else step.step_type.upper()
                    content = step.thought or step.output or ""
                    st.markdown(
                        f'<div class="step-card">{icon} <b>{label}</b>: {content[:200]}</div>',
                        unsafe_allow_html=True
                    )

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result.answer,
        "sources": result.sources,
        "steps":   [{"step_type": s.step_type, "tool": s.tool,
                     "thought": s.thought, "output": s.output}
                    for s in result.steps],
    })
