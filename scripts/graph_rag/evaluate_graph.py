"""
Retrieval evaluation for Graph RAG, identical protocol to
scripts/bm25/evaluate_bm25.py and scripts/vector_rag/evaluate_vector.py:
same 100 questions, same gold-passage -> chunk mapping (coverage >= 0.5),
same Precision@k / Recall@k / MRR / Full Evidence Recovery@k, k in {1,3,5,10}.

    python scripts/graph_rag/evaluate_graph.py

Outputs (results/graph/):
    graph_results.csv           one row per (retriever, question, k)
    graph_summary.csv           mean metrics per retriever x hop x k
    graph_report_table.csv      k = 10 table for the report
    graph_retrieval_details.csv matched entities + evidence paths (k = 10)
"""

import csv
import time

import pandas as pd

from graph_retrieval import (
    load_chunk_lookup,
    load_graph_system,
    retrieve_graph,
)
from kg_common import (
    EVALUATION_DIR,
    MAPPING_THRESHOLD,
    RESULTS_DIR,
    build_bm25_indexes,
    find_best_gold_mapping,
    get_question_document,
    load_evaluation,
)

import sys
from kg_common import BM25_DIR  # noqa: F401  (adds scripts/bm25 to sys.path)
from retrieval_metrics import calculate_retrieval_metrics  # noqa: E402

TOP_K_VALUES = [1, 3, 5, 10]

MODES = {
    "Graph": "graph",
    "Graph+BM25": "hybrid",
}


def get_relevant_chunks(question, indexes):

    document_id = get_question_document(question)

    if document_id not in indexes:
        return set()

    relevant = set()

    for gold_passage in question["gold_passages"]:
        best_chunk, score = find_best_gold_mapping(gold_passage, indexes[document_id]["chunks"])
        if best_chunk is not None and score >= MAPPING_THRESHOLD:
            relevant.add(best_chunk["chunk_id"])

    return relevant


def evaluate():

    print("Loading data ...")

    lookup = load_chunk_lookup()
    questions = load_evaluation(EVALUATION_DIR)
    indexes = build_bm25_indexes(list(lookup.values()))

    start = time.perf_counter()
    system = load_graph_system()
    load_time = time.perf_counter() - start

    print(f"Graph loaded in {load_time:.2f}s (extractor={system['extractor']})")

    rows, details = [], []

    for i, question in enumerate(questions, start=1):

        relevant_ids = get_relevant_chunks(question, indexes)

        for retriever, mode in MODES.items():

            t0 = time.perf_counter()
            results, info = retrieve_graph(
                question, system, lookup, top_k=max(TOP_K_VALUES), mode=mode,
                bm25_indexes=indexes, with_paths=False,
            )
            latency = time.perf_counter() - t0

            retrieved_ids = [r["chunk_id"] for r in results]

            for k in TOP_K_VALUES:
                metrics = calculate_retrieval_metrics(retrieved_ids, relevant_ids, k)
                rows.append({
                    "retriever": retriever,
                    "question_id": question["question_id"],
                    "hop_count": question["hop_count"],
                    "difficulty": question.get("difficulty_cell", ""),
                    "k": k,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "reciprocal_rank": metrics["reciprocal_rank"],
                    "evidence_recovery": metrics["evidence_recovery"],
                    "retrieval_latency_ms": latency * 1000,
                    "num_relevant_chunks": len(relevant_ids),
                    "seed_source": info["seed_source"],
                    "num_matched_entities": len(info["matched_entities"]),
                })

            # provenance paths for inspection (not timed)
            with_paths, _ = retrieve_graph(question, system, lookup, top_k=10, mode=mode,
                                           bm25_indexes=indexes, with_paths=True)
            details.append({
                "retriever": retriever,
                "question_id": question["question_id"],
                "hop_count": question["hop_count"],
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "matched_entities": " | ".join(info["matched_entities"]),
                "seed_source": info["seed_source"],
                "relevant_chunk_ids": " | ".join(sorted(relevant_ids)),
                "retrieved": " || ".join(
                    f"{r['rank']}:{r['chunk_id'].rsplit('::', 1)[-1]}"
                    f"{'*' if r['chunk_id'] in relevant_ids else ''} {r['evidence_path']}"
                    for r in with_paths
                ),
            })

        print(f"[{i}/{len(questions)}] {question['question_id']} | {question['hop_count']}-hop | "
              f"relevant={len(relevant_ids)} | seeds={info['seed_source']}")

    return rows, details, load_time


def summarise(df):

    grouped = (
        df.groupby(["retriever", "hop_count", "k"])
        .agg(precision=("precision", "mean"), recall=("recall", "mean"),
             mrr=("reciprocal_rank", "mean"), evidence_recovery=("evidence_recovery", "mean"),
             latency_ms=("retrieval_latency_ms", "mean"), questions=("question_id", "nunique"))
        .reset_index()
    )

    overall = (
        df.groupby(["retriever", "k"])
        .agg(precision=("precision", "mean"), recall=("recall", "mean"),
             mrr=("reciprocal_rank", "mean"), evidence_recovery=("evidence_recovery", "mean"),
             latency_ms=("retrieval_latency_ms", "mean"), questions=("question_id", "nunique"))
        .reset_index()
    )
    overall["hop_count"] = "Overall"

    grouped["hop_count"] = grouped["hop_count"].astype(str)
    return pd.concat([grouped, overall[grouped.columns]], ignore_index=True)


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, details, load_time = evaluate()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "graph_results.csv", index=False)

    with open(RESULTS_DIR / "graph_retrieval_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)

    summary = summarise(df)
    summary.to_csv(RESULTS_DIR / "graph_summary.csv", index=False)

    table = summary[summary["k"] == 10][["retriever", "hop_count", "precision", "recall", "mrr", "evidence_recovery"]].copy()
    table.columns = ["Retriever", "Reasoning Depth", "Precision@10", "Recall@10", "MRR", "Full Evidence Recovery@10"]
    table = table.round(3)
    table.to_csv(RESULTS_DIR / "graph_report_table.csv", index=False)

    seed_rate = df[df["k"] == 10].groupby("retriever")["seed_source"].apply(
        lambda s: (s.str.startswith("entities")).mean())

    print("\n" + "=" * 90)
    print("GRAPH RETRIEVAL - K = 10")
    print("=" * 90)
    print(table.to_string(index=False))
    print(f"\nQuestions with >= 1 matched graph entity: {seed_rate.to_dict()}")
    print(f"Graph load time: {load_time:.2f}s")
    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
