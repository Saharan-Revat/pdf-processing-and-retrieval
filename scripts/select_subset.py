import json
import os
import shutil
from collections import defaultdict, Counter
from pathlib import Path

from pypdf import PdfReader


PDF_ROOT = Path("dataset/CUAD_v1/full_contract_pdf")
EVAL_FILE = Path("dataset/nllp-cuad-multihop/dataset_eval.jsonl")
OUTPUT_DIR = Path("dataset/selected_pdfs")

MAX_PAGES = 30
MIN_QUESTIONS = 20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count questions and hops for each NLLP document
    stats = defaultdict(Counter)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            doc_id = (
                row["gold_passages"][0]["chunk_id"]
                .split("::")[0]
            )

            stats[doc_id][row["hop_count"]] += 1

    selected = []

    # Find corresponding CUAD PDFs
    for root, _, files in os.walk(PDF_ROOT):
        for filename in files:

            if not filename.lower().endswith(".pdf"):
                continue

            doc_id = os.path.splitext(filename)[0]

            if doc_id not in stats:
                continue

            source_path = Path(root) / filename

            try:
                pages = len(PdfReader(source_path).pages)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                continue

            total_questions = sum(stats[doc_id].values())

            # Frozen subset criteria
            if pages <= MAX_PAGES and total_questions >= MIN_QUESTIONS:

                destination = OUTPUT_DIR / filename
                shutil.copy2(source_path, destination)

                selected.append({
                    "doc_id": doc_id,
                    "pages": pages,
                    "hop_2": stats[doc_id][2],
                    "hop_3": stats[doc_id][3],
                    "hop_4": stats[doc_id][4],
                    "hop_5": stats[doc_id][5],
                    "total": total_questions,
                })

    selected.sort(key=lambda x: x["doc_id"])

    print(f"\nSelected {len(selected)} PDFs\n")

    for item in selected:
        print(
            f"{item['pages']:2} pages | "
            f"2-hop={item['hop_2']:2} | "
            f"3-hop={item['hop_3']:2} | "
            f"4-hop={item['hop_4']:2} | "
            f"5-hop={item['hop_5']:2} | "
            f"Total={item['total']:3} | "
            f"{item['doc_id']}"
        )

    print("\nTotal available questions:")
    for hop in [2, 3, 4, 5]:
        total = sum(item[f"hop_{hop}"] for item in selected)
        print(f"{hop}-hop: {total}")


if __name__ == "__main__":
    main()
