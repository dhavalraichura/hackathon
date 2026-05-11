# ─────────────────────────────────────────────
# llm.py — LLM interface
# Supports: Ollama (local/free), OpenAI, Anthropic
# ─────────────────────────────────────────────

import requests
import time
import config

# Maximum words per chunk sent to LLM — keeps prompt manageable for local models
MAX_CHUNK_WORDS = 200
# Maximum chunks to include in prompt
MAX_CHUNKS      = 3
# Maximum retries on empty/failed response
MAX_RETRIES     = 2


# ══════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════

def _build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build a concise RAG prompt.
    Truncates each chunk to MAX_CHUNK_WORDS and limits to MAX_CHUNKS
    to keep the total prompt size manageable for local models like Gemma4.
    """
    chunks = context_chunks[:MAX_CHUNKS]
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        words     = chunk["text"].split()
        truncated = " ".join(words[:MAX_CHUNK_WORDS])
        suffix    = "..." if len(words) > MAX_CHUNK_WORDS else ""
        context_lines.append(
            f"[Source {i}: {chunk['source']}]\n{truncated}{suffix}"
        )

    context = "\n\n---\n\n".join(context_lines)

    # Concise prompt — local models perform better with shorter, clearer instructions
    prompt = f"""You are an IT Knowledge Assistant. Answer the question using the sources below.
Cite sources as [Source: name]. If unsure, say so. Be concise and direct.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


def _build_summary_prompt(text: str) -> str:
    words     = text.split()
    truncated = " ".join(words[:600])
    return f"""Summarize the following in 3-5 bullet points. Be concise.

TEXT:
{truncated}

SUMMARY:"""


# ══════════════════════════════════════════════
# MAIN DISPATCHER
# ══════════════════════════════════════════════

def call_llm(query: str, context_chunks: list[dict]) -> str:
    """Call the configured LLM provider with a RAG prompt."""
    prompt = _build_prompt(query, context_chunks)

    if config.LLM_PROVIDER == "ollama":
        return _call_with_retry(_call_ollama, prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_with_retry(_call_openai, prompt)
    elif config.LLM_PROVIDER == "anthropic":
        return _call_with_retry(_call_anthropic, prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def summarize(text: str) -> str:
    """Summarize a long text (used by agent's summarizer tool)."""
    prompt = _build_summary_prompt(text)

    if config.LLM_PROVIDER == "ollama":
        return _call_with_retry(_call_ollama, prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_with_retry(_call_openai, prompt)
    elif config.LLM_PROVIDER == "anthropic":
        return _call_with_retry(_call_anthropic, prompt)


# ══════════════════════════════════════════════
# RETRY WRAPPER
# ══════════════════════════════════════════════

def _call_with_retry(fn, prompt: str) -> str:
    """
    Call fn(prompt) up to MAX_RETRIES times.
    Retries on empty response or error string.
    """
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = fn(prompt)
            if result and not result.startswith("⚠"):
                return result
            last_error = result
            if attempt < MAX_RETRIES:
                print(f"  ↻ LLM attempt {attempt} failed ({result[:60]}), retrying...")
                time.sleep(2)
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"  ↻ LLM attempt {attempt} exception: {e}, retrying...")
                time.sleep(2)

    return (
        last_error or
        "I was unable to generate a response. "
        "Please try rephrasing your question or ask something more specific."
    )


# ══════════════════════════════════════════════
# OLLAMA
# ══════════════════════════════════════════════

def _call_ollama(prompt: str) -> str:
    """
    Call local Ollama server using streaming mode.
    Streaming avoids connection timeouts on long responses —
    tokens are read as they arrive and reassembled into a full answer.
    """
    import json as _json

    url        = f"{config.OLLAMA_URL}/api/generate"
    word_count = len(prompt.split())
    print(f"  📤 Ollama prompt: {word_count} words | model: {config.LLM_MODEL}")

    try:
        resp = requests.post(
            url,
            json={
                "model":  config.LLM_MODEL,
                "prompt": prompt,
                "stream": True,          # stream tokens to avoid timeout
                "options": {
                    "temperature":    0.4,
                    "num_predict":    1024,  # max tokens to generate
                    "num_ctx":        4096,  # context window
                    "repeat_penalty": 1.1,   # reduce repetition
                }
            },
            stream=True,
            timeout=getattr(config, "OLLAMA_TIMEOUT", 180),
        )
        resp.raise_for_status()

        # Read streamed tokens and reassemble
        full_response = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = _json.loads(line)
                token = chunk.get("response", "")
                full_response.append(token)
                if chunk.get("done"):
                    break
            except _json.JSONDecodeError:
                continue

        answer = "".join(full_response).strip()

        if not answer:
            return "⚠ Ollama returned an empty response."

        print(f"  ✓ Ollama: {len(answer.split())} words returned")
        return answer

    except requests.exceptions.Timeout:
        return (
            "⚠ Ollama timed out. The prompt may be too long or the model is busy. "
            "Try asking a more specific question."
        )
    except requests.exceptions.ConnectionError:
        return "⚠ Ollama server not running. Start it with: ollama serve"
    except Exception as e:
        return f"⚠ Ollama error: {e}"


# ══════════════════════════════════════════════
# OPENAI
# ══════════════════════════════════════════════

def _call_openai(prompt: str) -> str:
    """Call OpenAI API (requires OPENAI_API_KEY in .env)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_KEY)
        resp   = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = resp.choices[0].message.content.strip()
        if not answer:
            return "⚠ OpenAI returned an empty response."
        return answer
    except Exception as e:
        return f"⚠ OpenAI error: {e}"


# ══════════════════════════════════════════════
# ANTHROPIC
# ══════════════════════════════════════════════

def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude API (requires ANTHROPIC_API_KEY in .env)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
        msg    = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = msg.content[0].text.strip()
        if not answer:
            return "⚠ Anthropic returned an empty response."
        return answer
    except Exception as e:
        return f"⚠ Anthropic error: {e}"