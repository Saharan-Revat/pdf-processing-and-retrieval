import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "results/vector/vector_summary.csv"
)

OUTPUT_FILE = Path(
    "results/vector/vector_report_table.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    k10 = df[
        df["k"] == 10
    ].copy()

    for column in [
        "precision",
        "recall",
        "mrr",
        "evidence_recovery"
    ]:
        k10[column] = (
            k10[column].round(3)
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

    print("\nVECTOR REPORT TABLE")
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