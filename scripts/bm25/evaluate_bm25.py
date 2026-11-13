import csv
import time
from pathlib import Path

from bm25_retrieval import (
    CHUNKS_DIR,
    EVALUATION_DIR,
    load_chunks,
    load_evaluation,
    build_bm25_indexes,
    retrieve_bm25,
    get_question_document,
    find_best_gold_mapping
)

from retrieval_metrics import calculate_retrieval_metrics

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_VALUES = [1, 3, 5, 10]

MAPPING_THRESHOLD = 0.50


def get_relevant_chunks(question, indexes):
    """
    Map NLLP gold passages to the project's new chunk IDs.

    Returns the unique set of project chunks considered
    relevant for this question.
    """

    document_id = get_question_document(question)

    if document_id not in indexes:
        return set()

    document_chunks = indexes[document_id]["chunks"]

    relevant_chunks = set()

    for gold_passage in question["gold_passages"]:

        best_chunk, score = find_best_gold_mapping(
            gold_passage,
            document_chunks
        )

        if best_chunk is not None and score >= MAPPING_THRESHOLD:
            relevant_chunks.add(
                best_chunk["chunk_id"]
            )

    return relevant_chunks




def evaluate():

    print("Loading data...")

    chunks = load_chunks(CHUNKS_DIR)
    questions = load_evaluation(EVALUATION_DIR)

    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")

    print("\nBuilding BM25 indexes...")

    start_index = time.perf_counter()

    indexes = build_bm25_indexes(chunks)

    index_time = time.perf_counter() - start_index

    print(
        f"Index construction time: "
        f"{index_time:.4f} seconds"
    )

    rows = []

    for i, question in enumerate(questions, start=1):

        relevant_ids = get_relevant_chunks(
            question,
            indexes
        )

        start_query = time.perf_counter()

        results = retrieve_bm25(
            question,
            indexes,
            top_k=max(TOP_K_VALUES)
        )

        latency = time.perf_counter() - start_query

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        for k in TOP_K_VALUES:

            metrics = calculate_retrieval_metrics(
                retrieved_ids,
                relevant_ids,
                k
            )

            rows.append({
                "question_id":
                    question["question_id"],

                "hop_count":
                    question["hop_count"],

                "difficulty":
                    question.get(
                        "difficulty_cell",
                        ""
                    ),

                "k": k,

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "reciprocal_rank":
                    metrics["reciprocal_rank"],

                "evidence_recovery":
                    metrics["evidence_recovery"],

                "retrieval_latency_ms":
                    latency * 1000,

                "num_relevant_chunks":
                    len(relevant_ids)
            })

        print(
            f"[{i}/{len(questions)}] "
            f"{question['question_id']} | "
            f"{question['hop_count']}-hop | "
            f"relevant={len(relevant_ids)}"
        )

    return rows, index_time


def save_results(rows):

    output_path = RESULTS_DIR / "bm25_results.csv"

    fieldnames = [
        "question_id",
        "hop_count",
        "difficulty",
        "k",
        "precision",
        "recall",
        "reciprocal_rank",
        "evidence_recovery",
        "retrieval_latency_ms",
        "num_relevant_chunks"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output_path}")



if __name__ == "__main__":

    rows, index_time = evaluate()

    save_results(rows)

    print(
        f"\nBM25 index construction time: "
        f"{index_time:.4f} seconds"
    )

    print("\nEvaluation complete.")