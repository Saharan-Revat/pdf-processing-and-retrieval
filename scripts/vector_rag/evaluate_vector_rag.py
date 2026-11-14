import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import csv
import re
import sys
import time
from pathlib import Path


from vector_retrieval import (
    load_vector_system,
    load_evaluation,
    retrieve_vector
)

from vector_rag import (
    load_generator,
    generate_answer
)


# Access gold-mapping utilities from BM25
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
BM25_DIR = SCRIPTS_DIR / "bm25"

sys.path.insert(0, str(BM25_DIR))

from bm25_retrieval import (
    CHUNKS_DIR,
    load_chunks,
    build_bm25_indexes,
    get_question_document,
    find_best_gold_mapping
)


RESULTS_DIR = Path("results/vector")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "vector_rag_results.csv"

TOP_K = 5
MAPPING_THRESHOLD = 0.50
QUESTIONS_PER_HOP = 25


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_match(generated, gold):
    """
    Relaxed answer matching.

    The complete gold answer must appear in the
    generated response.
    """

    generated = normalize_answer(generated)
    gold = normalize_answer(gold)

    if not generated or not gold:
        return False

    return gold in generated


def select_balanced_sample(questions):
    """
    Select first 2 questions from each hop level.
    """

    selected = []

    for hop in [2, 3, 4, 5]:

        hop_questions = [
            q for q in questions
            if q["hop_count"] == hop
        ]

        selected.extend(
            hop_questions[:QUESTIONS_PER_HOP]
        )

    return selected


def get_relevant_chunks(
    question,
    mapping_indexes
):
    document_id = get_question_document(question)

    if document_id not in mapping_indexes:
        return set()

    document_chunks = (
        mapping_indexes[document_id]["chunks"]
    )

    relevant = set()

    for gold_passage in question["gold_passages"]:

        best_chunk, score = find_best_gold_mapping(
            gold_passage,
            document_chunks
        )

        if (
            best_chunk is not None
            and score >= MAPPING_THRESHOLD
        ):
            relevant.add(
                best_chunk["chunk_id"]
            )

    return relevant


def evaluate():

    print("Loading data...")

    questions = load_evaluation()
    chunks = load_chunks(CHUNKS_DIR)

    sample = select_balanced_sample(
        questions
    )

    print(
        f"Selected {len(sample)} questions."
    )

    # Gold mapping structure
    mapping_indexes = build_bm25_indexes(
        chunks
    )

    # Vector retrieval
    index, metadata, embedding_model = (
        load_vector_system()
    )

    # Generator
    tokenizer, generation_model = (
        load_generator()
    )

    rows = []

    for i, question in enumerate(
        sample,
        start=1
    ):

        print("\n" + "=" * 80)
        print(
            f"[{i}/{len(sample)}] "
            f"{question['question_id']} "
            f"({question['hop_count']}-hop)"
        )
        print("=" * 80)

        print("\nQuestion:")
        print(question["question"])

        print("\nGold:")
        print(question["gold_answer"])

        relevant_ids = get_relevant_chunks(
            question,
            mapping_indexes
        )

        start = time.perf_counter()

        retrieved = retrieve_vector(
            question,
            index,
            metadata,
            embedding_model,
            top_k=TOP_K,
            restrict_to_document=True
        )

        retrieval_time = (
            time.perf_counter() - start
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in retrieved
        ]

        retrieved_relevant = (
            set(retrieved_ids)
            .intersection(relevant_ids)
        )

        evidence_retrieved = (
            len(retrieved_relevant) > 0
        )

        full_evidence_retrieved = (
            bool(relevant_ids)
            and relevant_ids.issubset(
                set(retrieved_ids)
            )
        )

        generation_start = (
            time.perf_counter()
        )

        generated = generate_answer(
            question["question"],
            retrieved,
            tokenizer,
            generation_model
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        matched = answer_match(
            generated,
            question["gold_answer"]
        )

        print("\nGenerated:")
        print(generated)

        print("\nRetrieved chunks:")

        for result in retrieved:
            marker = (
                "RELEVANT"
                if result["chunk_id"]
                in relevant_ids
                else "-"
            )

            print(
                f"  Rank {result['rank']} | "
                f"{marker} | "
                f"{result['chunk_id']}"
            )

        print(
            f"\nAny gold evidence retrieved: "
            f"{evidence_retrieved}"
        )

        print(
            f"Full gold evidence retrieved: "
            f"{full_evidence_retrieved}"
        )

        print(
            f"Answer match: {matched}"
        )

        rows.append({
            "question_id":
                question["question_id"],

            "hop_count":
                question["hop_count"],

            "question":
                question["question"],

            "gold_answer":
                question["gold_answer"],

            "generated_answer":
                generated,

            "answer_match":
                matched,

            "any_evidence_retrieved":
                evidence_retrieved,

            "full_evidence_retrieved":
                full_evidence_retrieved,

            "num_relevant_chunks":
                len(relevant_ids),

            "num_relevant_retrieved":
                len(retrieved_relevant),

            "retrieved_chunk_ids":
                " | ".join(retrieved_ids),

            "retrieval_latency_ms":
                retrieval_time * 1000,

            "generation_latency_ms":
                generation_time * 1000
        })

    return rows


def save_results(rows):

    if not rows:
        return

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved results to {OUTPUT_FILE}"
    )


def print_summary(rows):

    total = len(rows)

    matched = sum(
        row["answer_match"]
        for row in rows
    )

    any_evidence = sum(
        row["any_evidence_retrieved"]
        for row in rows
    )

    full_evidence = sum(
        row["full_evidence_retrieved"]
        for row in rows
    )

    print("\n" + "=" * 80)
    print("VECTOR RAG RESULTS SUMMARY")
    print("=" * 80)

    print(
        f"Questions: {total}"
    )

    print(
        f"Answer matches: "
        f"{matched}/{total} "
        f"({matched / total:.1%})"
    )

    print(
        f"Any gold evidence retrieved: "
        f"{any_evidence}/{total} "
        f"({any_evidence / total:.1%})"
    )

    print(
        f"Full gold evidence retrieved: "
        f"{full_evidence}/{total} "
        f"({full_evidence / total:.1%})"
    )


if __name__ == "__main__":

    rows = evaluate()

    save_results(rows)

    print_summary(rows)