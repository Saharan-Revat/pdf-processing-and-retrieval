"""
End-to-end Graph RAG question answering.

Uses exactly the same answer generator, prompt, evidence budget (top-5
chunks), question sample and answer-matching rule as
scripts/vector_rag/evaluate_vector_rag.py, so that the retrieval
architecture is the only experimental variable.

    python scripts/graph_rag/evaluate_graph_rag.py              # Graph and Graph+BM25
    python scripts/graph_rag/evaluate_graph_rag.py --mode graph

Needs the same environment as Vector RAG (transformers, torch,
sentence-transformers, faiss) because it imports the shared generator from
scripts/vector_rag/vector_rag.py.

Output: results/graph/graph_rag_results.csv, graph_rag_summary.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from graph_retrieval import load_chunk_lookup, load_graph_system, retrieve_graph
from kg_common import EVALUATION_DIR, RESULTS_DIR, build_bm25_indexes, load_evaluation
from evaluate_graph import get_relevant_chunks

VECTOR_DIR = Path(__file__).resolve().parents[1] / "vector_rag"
sys.path.insert(0, str(VECTOR_DIR))

from vector_rag import load_generator, generate_answer  # noqa: E402
from evaluate_vector_rag import answer_match, select_balanced_sample, TOP_K  # noqa: E402

MODES = {"graph": "Graph RAG", "hybrid": "Graph+BM25 RAG"}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["graph", "hybrid", "both"], default="both")
    args = parser.parse_args()

    modes = ["graph", "hybrid"] if args.mode == "both" else [args.mode]

    lookup = load_chunk_lookup()
    indexes = build_bm25_indexes(list(lookup.values()))
    system = load_graph_system()
    sample = select_balanced_sample(load_evaluation(EVALUATION_DIR))
    tokenizer, model = load_generator()

    rows = []

    for i, question in enumerate(sample, start=1):

        relevant = get_relevant_chunks(question, indexes)

        for mode in modes:

            t0 = time.perf_counter()
            retrieved, info = retrieve_graph(question, system, lookup, top_k=TOP_K, mode=mode,
                                             bm25_indexes=indexes, with_paths=True)
            retrieval_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            generated = generate_answer(question["question"], retrieved, tokenizer, model)
            generation_ms = (time.perf_counter() - t1) * 1000

            ids = [r["chunk_id"] for r in retrieved]
            hit = set(ids) & relevant

            rows.append({
                "retriever": MODES[mode],
                "question_id": question["question_id"],
                "hop_count": question["hop_count"],
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "generated_answer": generated,
                "answer_match": answer_match(generated, question["gold_answer"]),
                "any_evidence_retrieved": bool(hit),
                "full_evidence_retrieved": bool(relevant) and relevant.issubset(ids),
                "num_relevant_chunks": len(relevant),
                "num_relevant_retrieved": len(hit),
                "matched_entities": " | ".join(info["matched_entities"]),
                "retrieved_chunk_ids": " | ".join(ids),
                "evidence_paths": " || ".join(r["evidence_path"] for r in retrieved),
                "source_pages": " | ".join(str(r["pages"]) for r in retrieved),
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": generation_ms,
            })

            print(f"[{i}/{len(sample)}] {MODES[mode]:15s} {question['hop_count']}-hop "
                  f"match={rows[-1]['answer_match']!s:5} | {generated[:60]!r} | gold={question['gold_answer'][:40]!r}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "graph_rag_results.csv", index=False)

    def agg(frame):
        return frame.agg(
            questions=("question_id", "nunique"),
            correct_answers=("answer_match", "sum"),
            answer_accuracy=("answer_match", "mean"),
            any_evidence_rate=("any_evidence_retrieved", "mean"),
            full_evidence_rate=("full_evidence_retrieved", "mean"),
            avg_retrieval_latency_ms=("retrieval_latency_ms", "mean"),
            avg_generation_latency_ms=("generation_latency_ms", "mean"),
        ).reset_index()

    by_hop = agg(df.groupby(["retriever", "hop_count"]))
    by_hop["hop_count"] = by_hop["hop_count"].astype(str)
    overall = agg(df.groupby("retriever"))
    overall["hop_count"] = "Overall"
    summary = pd.concat([by_hop, overall[by_hop.columns]], ignore_index=True)
    summary.to_csv(RESULTS_DIR / "graph_rag_summary.csv", index=False)

    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
