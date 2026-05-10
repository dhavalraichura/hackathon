# ─────────────────────────────────────────────
# llm.py — LLM interface
# Supports: Ollama (local/free), OpenAI, Anthropic
# ─────────────────────────────────────────────

import requests
import config


def _build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build a RAG prompt from retrieved chunks."""
    context_lines = []
    for i, chunk in enumerate(context_chunks, 1):
        context_lines.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_lines)

    prompt = f"""You are an Enterprise Knowledge Assistant for an IT services company.
Answer the employee's question using ONLY the context provided below.
Always cite which source(s) your answer comes from.
If the context does not contain enough information, say: "I don't have enough information to answer this confidently.
        TASK:
        1. Answer the user's question using the context.
        2. Check the files uploaded for any relevant information that can help answer the question.
        3. If the question is about contact details, names, or specific data, prioritize extracting that from the files.
        4. Also look at the files to see if they contain any relevant information that could help the user find the answer, even if the answer isn't directly in the files. For example, if the user asks "Who is the CEO?" and the file contains a company org chart but doesn't explicitly say "CEO", you can infer the answer based on titles and structure.
        5. The file names and types are provided in the context. Use that to understand the source of information.
        6. If you don't know the answer, say you don't know, but also mention if the files contain any relevant information that could help the user find the answer
        7. If the question is vague, use the context to ask a clarifying question back to the user."

CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}

INSTRUCTIONS:
- Answer clearly and concisely
- Cite sources like: [Source: docker-intro]
- If unsure, say so — do not hallucinate
- If steps are involved, use numbered list

ANSWER:"""
    return prompt


def call_llm(query: str, context_chunks: list[dict]) -> str:
    """
    Call the configured LLM provider with RAG prompt.
    Provider set in config.LLM_PROVIDER
    """
    prompt = _build_prompt(query, context_chunks)

    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    elif config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _call_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model":  config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 512,
                }
            },
            timeout=config.OLLAMA_TIMEOUT   # instead of hardcoded 120
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()

        # ← ADD THIS: catch silent empty responses
        if not answer:
            return "I found relevant sources but was unable to generate a response. Please try rephrasing your question."

        return answer
    except requests.exceptions.Timeout:
        return "⚠ Ollama timed out. The model may be overloaded — try again in a moment."
    except requests.exceptions.ConnectionError:
        return "⚠ Ollama server not running. Start it with: ollama serve"
    except Exception as e:
        return f"⚠ Ollama error: {e}"


def _call_openai(prompt: str) -> str:
    """Call OpenAI API (requires OPENAI_API_KEY in .env)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ OpenAI error: {e}"


def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude API (requires ANTHROPIC_API_KEY in .env)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"⚠ Anthropic error: {e}"


def summarize(text: str) -> str:
    """Summarize a long text (Tool 3 of agent)."""
    prompt = f"""Summarize the following text in 3-5 bullet points. Be concise and factual.

TEXT:
{text[:3000]}

SUMMARY:"""

    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    elif config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
