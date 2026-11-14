import pandas as pd
from pathlib import Path


BM25_FILE = Path(
    "results/bm25_summary.csv"
)

VECTOR_FILE = Path(
    "results/vector/vector_summary.csv"
)

OUTPUT_FILE = Path(
    "results/retriever_comparison.csv"
)


def main():

    bm25 = pd.read_csv(BM25_FILE)
    vector = pd.read_csv(VECTOR_FILE)

    bm25 = bm25[
        bm25["k"] == 10
    ].copy()

    vector = vector[
        vector["k"] == 10
    ].copy()

    columns = [
        "hop_count",
        "precision",
        "recall",
        "mrr",
        "evidence_recovery",
        "latency_ms"
    ]

    bm25 = bm25[columns]
    vector = vector[columns]

    bm25["retriever"] = "BM25"
    vector["retriever"] = "Vector"

    comparison = pd.concat(
        [bm25, vector],
        ignore_index=True
    )

    comparison = comparison[
        [
            "retriever",
            "hop_count",
            "precision",
            "recall",
            "mrr",
            "evidence_recovery",
            "latency_ms"
        ]
    ]

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nBM25 VS VECTOR — K=10")
    print("=" * 90)

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nSaved comparison to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()