import pandas as pd
from pathlib import Path


INPUT_FILE = Path("results/bm25_summary.csv")
OUTPUT_FILE = Path("results/bm25_report_table.csv")


def main():

    df = pd.read_csv(INPUT_FILE)

    # We use K=10 for the main hop comparison
    k10 = df[df["k"] == 10].copy()

    k10["precision"] = k10["precision"].round(3)
    k10["recall"] = k10["recall"].round(3)
    k10["mrr"] = k10["mrr"].round(3)
    k10["evidence_recovery"] = (
        k10["evidence_recovery"].round(3)
    )

    table = k10[
        [
            "hop_count",
            "precision",
            "recall",
            "mrr",
            "evidence_recovery"
        ]
    ].copy()

    table.columns = [
        "Reasoning Depth",
        "Precision@10",
        "Recall@10",
        "MRR",
        "Full Evidence Recovery@10"
    ]

    table.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nBM25 REPORT TABLE")
    print("=" * 75)

    print(
        table.to_string(
            index=False
        )
    )

    print(
        f"\nSaved report table to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()