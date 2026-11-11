import json
import random
from collections import defaultdict, Counter
from pathlib import Path

EVAL_FILE = Path("dataset/nllp-cuad-multihop/dataset_eval.jsonl")
SELECTED_PDFS = Path("dataset/selected_pdfs")
OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "evaluation_questions.jsonl"

QUESTIONS_PER_HOP = 25
SEED = 42

random.seed(SEED)


def get_doc_id(row):
    return row["gold_passages"][0]["chunk_id"].split("::")[0]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Our selected 17 contracts
    selected_docs = {
        p.stem
        for p in SELECTED_PDFS.iterdir()
        if p.suffix.lower() == ".pdf"
    }

    print(f"Selected PDFs: {len(selected_docs)}")

    # Group eligible questions by hop -> document
    questions = defaultdict(lambda: defaultdict(list))

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = get_doc_id(row)

            if doc_id in selected_docs:
                questions[row["hop_count"]][doc_id].append(row)

    final_questions = []

    for hop in [2, 3, 4, 5]:
        pools = questions[hop]

        # Shuffle questions inside each contract reproducibly
        for doc_id in pools:
            random.shuffle(pools[doc_id])

        # Shuffle document order so the same contracts
        # aren't always favored
        docs = list(pools.keys())
        random.shuffle(docs)

        selected = []

        # Round-robin across contracts
        while len(selected) < QUESTIONS_PER_HOP:
            added_this_round = False

            for doc_id in docs:
                if len(selected) >= QUESTIONS_PER_HOP:
                    break

                if pools[doc_id]:
                    selected.append(pools[doc_id].pop())
                    added_this_round = True

            if not added_this_round:
                break

        if len(selected) != QUESTIONS_PER_HOP:
            raise ValueError(
                f"Could only select {len(selected)} questions "
                f"for {hop}-hop."
            )

        final_questions.extend(selected)

    # Final deterministic shuffle
    random.shuffle(final_questions)

    # Check duplicates
    ids = [q["question_id"] for q in final_questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs detected.")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in final_questions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- Summary ----------
    hop_counts = Counter()
    doc_counts = Counter()

    for row in final_questions:
        hop_counts[row["hop_count"]] += 1
        doc_counts[get_doc_id(row)] += 1

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Total questions: {len(final_questions)}")

    print("\nHop distribution:")
    for hop in [2, 3, 4, 5]:
        print(f"{hop}-hop: {hop_counts[hop]}")

    print(f"\nContracts represented: {len(doc_counts)}/{len(selected_docs)}")

    print("\nQuestions per contract:")
    for doc, count in sorted(
        doc_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):
        print(f"{count:2} | {doc}")

    missing = selected_docs - set(doc_counts)

    if missing:
        print("\nWARNING - contracts not represented:")
        for doc in sorted(missing):
            print(doc)


if __name__ == "__main__":
    main()