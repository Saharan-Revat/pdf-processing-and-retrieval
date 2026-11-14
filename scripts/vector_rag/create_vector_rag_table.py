import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "results/vector/vector_rag_summary.csv"
)

OUTPUT_FILE = Path(
    "results/vector/vector_rag_report_table.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    table = df[
        [
            "hop_count",
            "questions",
            "correct_answers",
            "answer_accuracy",
            "any_evidence_rate",
            "full_evidence_rate"
        ]
    ].copy()

    table["answer_accuracy"] = (
        table["answer_accuracy"] * 100
    ).round(1)

    table["any_evidence_rate"] = (
        table["any_evidence_rate"] * 100
    ).round(1)

    table["full_evidence_rate"] = (
        table["full_evidence_rate"] * 100
    ).round(1)

    table.columns = [
        "Reasoning Depth",
        "Questions",
        "Correct Answers",
        "Gold Answer Match (%)",
        "Any Evidence Retrieved (%)",
        "Full Evidence Retrieved (%)"
    ]

    table.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nVECTOR RAG REPORT TABLE")
    print("=" * 100)

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