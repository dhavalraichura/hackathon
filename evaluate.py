# ─────────────────────────────────────────────
# evaluate.py — Evaluation framework
# Metrics: Precision, Recall, F1, Semantic Similarity, LLM-as-Judge
# ─────────────────────────────────────────────

import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

import config
from retriever import Retriever
from llm import call_llm


# ── Sample Evaluation Dataset ─────────────────
# Format: question, expected_sources (partial match), expected_answer_keywords
SAMPLE_EVAL_DATASET = [
    {
        "question": "How do I restart a Kubernetes pod?",
        "relevant_sources": ["kubernetes-pods"],        # matches _get_sample_chunks()
        "answer_keywords": ["kubectl", "delete", "pod"]
    },
    {
        "question": "How do I set up a Kafka consumer in Python?",
        "relevant_sources": ["kafka-consumer"],
        "answer_keywords": ["confluent-kafka", "subscribe", "poll"]
    },
    {
        "question": "What is Docker and what are its benefits?",
        "relevant_sources": ["docker-intro"],
        "answer_keywords": ["container", "application", "infrastructure"]
    },
    {
        "question": "How do I create a Python virtual environment?",
        "relevant_sources": ["python-venv"],
        "answer_keywords": ["venv", "activate", "pip"]
    },
    {
        "question": "Steps to onboard a new engineer",
        "relevant_sources": ["onboarding-sop", "onboarding_sop.txt"],  # both variants
        "answer_keywords": ["VPN", "repository", "Docker"]
    },
    {
        "question": "My Docker container is restarting with exit code 137",
        "relevant_sources": ["ticket-002", "Ticket #2"],
        "answer_keywords": ["memory", "OOM", "mem_limit"]
    },
    {
        "question": "How to fix database connection refused on port 5432?",
        "relevant_sources": ["ticket-001", "Ticket #1"],
        "answer_keywords": ["PostgreSQL", "systemctl", "firewall"]
    },
    {
        "question": "How do I run a FastAPI application?",
        "relevant_sources": ["fastapi-setup"],
        "answer_keywords": ["uvicorn", "main:app", "pip install"]
    },
]

class Evaluator:
    """
    Evaluates the RAG system on a test dataset.
    Computes: Precision@k, Recall@k, F1, Semantic Similarity, LLM-as-Judge
    """

    def __init__(self, use_reranker: bool = False):
        self.retriever  = Retriever(use_reranker=use_reranker)
        self.embedder   = SentenceTransformer(config.EMBEDDING_MODEL)
        self.eval_data  = self._load_eval_data()

    def _load_eval_data(self) -> list:
        """Load eval dataset from file or use sample."""
        try:
            with open(config.EVAL_DATASET_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            return SAMPLE_EVAL_DATASET

    # ── Retrieval Metrics ─────────────────────

    def compute_retrieval_metrics(self) -> dict:
        """
        Compute Precision@k, Recall@k, F1@k for all eval questions.

        Precision@k = relevant retrieved / k
        Recall@k    = relevant retrieved / total relevant
        F1@k        = harmonic mean of Precision and Recall
        """
        print("\n📐 Computing Retrieval Metrics...")
        precisions, recalls, f1s = [], [], []

        for item in self.eval_data:
            question         = item["question"]
            relevant_sources = item["relevant_sources"]

            # Retrieve
            results = self.retriever.retrieve(question, top_k=config.TOP_K)

            # Check which retrieved sources are relevant
            retrieved_relevant = 0
            for r in results:
                if any(rel in r["source"] for rel in relevant_sources):
                    retrieved_relevant += 1

            k          = len(results)
            total_rel  = len(relevant_sources)

            prec = retrieved_relevant / k if k > 0 else 0.0
            rec  = retrieved_relevant / total_rel if total_rel > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        return {
            "precision": round(np.mean(precisions), 4),
            "recall":    round(np.mean(recalls), 4),
            "f1":        round(np.mean(f1s), 4),
            "per_question": [
                {
                    "question":  item["question"][:60],
                    "precision": round(p, 3),
                    "recall":    round(r, 3),
                    "f1":        round(f, 3),
                }
                for item, p, r, f in zip(self.eval_data, precisions, recalls, f1s)
            ]
        }

    # ── Semantic Similarity ───────────────────

    def compute_semantic_similarity(self) -> dict:
        """
        Compute cosine similarity between LLM answer and expected keywords.
        Higher = answer is semantically closer to expected answer.
        """
        print("🧮 Computing Semantic Similarity...")
        similarities = []

        for item in self.eval_data:
            question = item["question"]
            expected = " ".join(item["answer_keywords"])

            # Get answer from LLM
            results = self.retriever.retrieve(question)
            if not results:
                similarities.append(0.0)
                continue

            answer = call_llm(question, results)

            # Embed both
            vecs = self.embedder.encode([answer, expected])
            sim  = float(np.dot(vecs[0], vecs[1]) /
                         (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9))
            similarities.append(max(0.0, sim))

        return {
            "semantic_similarity": round(np.mean(similarities), 4),
            "per_question": [
                {"question": item["question"][:60], "similarity": round(s, 3)}
                for item, s in zip(self.eval_data, similarities)
            ]
        }

    # ── LLM-as-Judge ─────────────────────────

    def llm_as_judge(self, question: str, answer: str, expected_keywords: list) -> dict:
        """
        Use LLM to score an answer on Accuracy, Completeness, Relevance (1-5 each).
        """
        judge_prompt = f"""You are an expert evaluator. Score the following answer on 3 dimensions.
Return ONLY valid JSON with keys: accuracy, completeness, relevance (each 1-5).

QUESTION: {question}
EXPECTED KEYWORDS: {', '.join(expected_keywords)}
ANSWER: {answer}

Score 5 = perfect, 1 = completely wrong.
JSON only, no explanation:"""

        from llm import _call_ollama, _call_openai, _call_anthropic
        if config.LLM_PROVIDER == "ollama":
            raw = _call_ollama(judge_prompt)
        elif config.LLM_PROVIDER == "openai":
            raw = _call_openai(judge_prompt)
        else:
            raw = _call_anthropic(judge_prompt)

        try:
            import re
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return {
                    "accuracy":     int(scores.get("accuracy", 3)),
                    "completeness": int(scores.get("completeness", 3)),
                    "relevance":    int(scores.get("relevance", 3)),
                }
        except Exception:
            pass
        return {"accuracy": 3, "completeness": 3, "relevance": 3}

    # ── Full Evaluation Report ────────────────

    def run_full_evaluation(self, include_llm_judge: bool = False) -> dict:
        """Run all evaluation metrics and print a report."""
        print("\n" + "="*60)
        print("  📊 ENTERPRISE KNOWLEDGE ASSISTANT — EVALUATION REPORT")
        print("="*60)

        results = {}

        # 1. Retrieval metrics
        ret_metrics = self.compute_retrieval_metrics()
        results["retrieval"] = ret_metrics

        print("\n📐 RETRIEVAL METRICS (averaged over test set)")
        print(tabulate([
            ["Precision@k", ret_metrics["precision"], "≥ 0.70 target"],
            ["Recall@k",    ret_metrics["recall"],    "≥ 0.65 target"],
            ["F1 Score",    ret_metrics["f1"],         "≥ 0.75 target"],
        ], headers=["Metric", "Score", "Target"], tablefmt="rounded_outline"))

        print("\n📋 Per-Question Breakdown:")
        print(tabulate(
            [[q["question"], q["precision"], q["recall"], q["f1"]]
             for q in ret_metrics["per_question"]],
            headers=["Question", "Precision", "Recall", "F1"],
            tablefmt="rounded_outline"
        ))

        # 2. Semantic similarity (optional — calls LLM for each question)
        if include_llm_judge:
            sem = self.compute_semantic_similarity()
            results["semantic"] = sem
            print(f"\n🧮 SEMANTIC SIMILARITY: {sem['semantic_similarity']}")

        # 3. Summary verdict
        f1 = ret_metrics["f1"]
        if f1 >= 0.75:
            verdict = "✅ EXCELLENT — Ready for demo"
        elif f1 >= 0.60:
            verdict = "🟡 GOOD — Minor improvements possible"
        else:
            verdict = "🔴 NEEDS IMPROVEMENT — Check chunking & embedding"

        print(f"\n🏁 VERDICT: {verdict}")
        print("="*60 + "\n")

        return results


# ── CLI entry point ───────────────────────────
if __name__ == "__main__":
    import sys
    llm_judge = "--llm-judge" in sys.argv
    ev = Evaluator()
    ev.run_full_evaluation(include_llm_judge=llm_judge)
