# ─────────────────────────────────────────────
# agent.py — ReAct Agentic Engine
# Tools: doc_search, ticket_lookup, summarizer, escalate
# Pattern: Reason → Act → Observe → loop
# ─────────────────────────────────────────────

import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

import config
from retriever import Retriever
from llm import call_llm, summarize


@dataclass
class AgentStep:
    """Represents one step in the ReAct loop."""
    step_type: str          # "think" | "act" | "observe"
    tool:      str = ""     # tool called
    input:     str = ""     # tool input
    output:    str = ""     # tool output
    thought:   str = ""     # agent's reasoning


@dataclass
class AgentResult:
    """Final result from agent run."""
    answer:    str
    sources:   list[dict]
    steps:     list[AgentStep]
    confident: bool
    escalated: bool = False


class EnterpriseAgent:
    """
    ReAct agent with 4 tools:
      Tool 1: doc_search     — semantic search over vector DB
      Tool 2: ticket_lookup  — keyword search over CSV tickets
      Tool 3: summarizer     — summarize long retrieved content
      Tool 4: escalate       — flag query for human review
    """

    def __init__(self, use_reranker: bool = False):
        self.retriever = Retriever(use_reranker=use_reranker)
        self.tickets_df = self._load_tickets()

    # ── Tool Implementations ──────────────────

    def tool_doc_search(self, query: str) -> tuple[list[dict], str]:
        """Tool 1: Search documents in vector DB."""
        results = self.retriever.retrieve(query, top_k=config.TOP_K)
        if not results:
            return [], "No relevant documents found."
        summary = f"Found {len(results)} relevant chunks:\n"
        for i, r in enumerate(results, 1):
            summary += f"  [{i}] {r['source']} (score: {r['score']})\n"
        return results, summary

    def tool_ticket_lookup(self, query: str) -> tuple[list[dict], str]:
        """Tool 2: Keyword search over support tickets CSV."""
        if self.tickets_df is None or self.tickets_df.empty:
            return [], "No ticket database available."

        query_lower = query.lower()
        keywords = query_lower.split()

        # Simple keyword matching across all text columns
        mask = pd.Series([False] * len(self.tickets_df))
        for col in self.tickets_df.columns:
            col_text = self.tickets_df[col].astype(str).str.lower()
            for kw in keywords:
                mask = mask | col_text.str.contains(kw, na=False)

        matches = self.tickets_df[mask].head(3)
        if matches.empty:
            return [], "No matching tickets found."

        ticket_chunks = []
        for _, row in matches.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in matches.columns])
            ticket_chunks.append({
                "text":   text,
                "source": f"Ticket #{row.get('id', 'N/A')}",
                "score":  0.7,
            })

        summary = f"Found {len(ticket_chunks)} related tickets."
        return ticket_chunks, summary

    def tool_summarizer(self, chunks: list[dict]) -> str:
        """Tool 3: Summarize retrieved chunks."""
        combined = "\n\n".join([c["text"] for c in chunks])
        return summarize(combined)

    def tool_escalate(self, query: str, reason: str) -> str:
        """Tool 4: Escalate to human when confidence is low."""
        return (
            f"🚨 This query has been escalated for human review.\n"
            f"Reason: {reason}\n"
            f"Query: {query}\n\n"
            f"Please contact your IT helpdesk or team lead for assistance."
        )

    # ── ReAct Loop ───────────────────────────

    def run(self, query: str) -> AgentResult:
        """
        Main ReAct loop:
        1. THINK  — reason about what to do
        2. ACT    — call a tool
        3. OBSERVE— process result, decide next step
        4. Repeat until answer ready or escalate
        """
        steps      = []
        all_chunks = []
        max_steps  = 4

        for step_num in range(max_steps):

            # ── THINK ──
            thought = self._think(query, all_chunks, step_num)
            steps.append(AgentStep(
                step_type="think",
                thought=thought
            ))

            # ── DECIDE TOOL ──
            tool_name = self._decide_tool(query, all_chunks, step_num)

            # ── ACT ──
            if tool_name == "doc_search":
                chunks, obs = self.tool_doc_search(query)
                all_chunks.extend(chunks)
                steps.append(AgentStep(
                    step_type="act",
                    tool="doc_search",
                    input=query,
                    output=obs
                ))

            elif tool_name == "ticket_lookup":
                chunks, obs = self.tool_ticket_lookup(query)
                all_chunks.extend(chunks)
                steps.append(AgentStep(
                    step_type="act",
                    tool="ticket_lookup",
                    input=query,
                    output=obs
                ))

            elif tool_name == "summarizer":
                if all_chunks:
                    obs = self.tool_summarizer(all_chunks)
                    steps.append(AgentStep(
                        step_type="act",
                        tool="summarizer",
                        input=f"{len(all_chunks)} chunks",
                        output=obs[:500]
                    ))

            elif tool_name == "answer":
                # Ready to generate final answer
                break

            elif tool_name == "escalate":
                answer = self.tool_escalate(query, "Low confidence in retrieved results")
                return AgentResult(
                    answer=answer,
                    sources=[],
                    steps=steps,
                    confident=False,
                    escalated=True
                )

        # ── GENERATE ANSWER ──
        if not all_chunks:
            return AgentResult(
                answer=self.tool_escalate(query, "No relevant documents found"),
                sources=[],
                steps=steps,
                confident=False,
                escalated=True
            )

        # Deduplicate chunks
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            if c["text"] not in seen:
                seen.add(c["text"])
                unique_chunks.append(c)

        # Check confidence
        confident = self.retriever.is_confident(unique_chunks[:config.TOP_K])

        if not confident:
            # Low confidence — escalate
            answer = (
                self.tool_escalate(query, "Retrieved documents have low relevance scores")
                + "\n\nHere is my best attempt based on available information:\n\n"
                + call_llm(query, unique_chunks[:config.TOP_K])
            )
            return AgentResult(
                answer=answer,
                sources=unique_chunks[:config.TOP_K],
                steps=steps,
                confident=False,
                escalated=True
            )

        answer = call_llm(query, unique_chunks[:config.TOP_K])

        return AgentResult(
            answer=answer,
            sources=unique_chunks[:config.TOP_K],
            steps=steps,
            confident=confident,
            escalated=False
        )

    # ── Internal Helpers ──────────────────────

    def _think(self, query: str, chunks_so_far: list, step: int) -> str:
        """Generate agent thought for current step."""
        if step == 0:
            return f"I need to find information about: '{query}'. I'll start by searching the document database."
        elif step == 1 and not chunks_so_far:
            return "Document search returned nothing. Let me check the ticket system."
        elif step == 1:
            return f"Found {len(chunks_so_far)} chunks. Let me also check tickets for related issues."
        elif step == 2:
            return f"I now have {len(chunks_so_far)} total context chunks. I have enough to generate an answer."
        else:
            return "Generating final answer from collected context."

    def _decide_tool(self, query: str, chunks_so_far: list, step: int) -> str:
        """Decide which tool to call based on current step and context."""
        query_lower = query.lower()
        is_ticket_query = any(w in query_lower for w in [
            "ticket", "issue", "error", "problem", "fix", "resolve",
            "crash", "fail", "broken", "not working"
        ])

        if step == 0:
            return "doc_search"
        elif step == 1:
            if is_ticket_query or not chunks_so_far:
                return "ticket_lookup"
            return "answer"
        elif step == 2:
            if len(chunks_so_far) > 5:
                return "summarizer"
            return "answer"
        else:
            if not chunks_so_far:
                return "escalate"
            return "answer"

    def _load_tickets(self):
        """Load tickets CSV if available."""
        path = Path(config.TICKETS_CSV)
        if path.exists():
            try:
                return pd.read_csv(str(path))
            except Exception:
                pass
        return None
