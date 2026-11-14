import pandas as pd
from pathlib import Path


BM25_FILE = Path(
    "results/bm25_summary.csv"
)

VECTOR_FILE = Path(
    "results/vector/vector_summary.csv"
)

GRAPH_FILE = Path(
    "results/graph/graph_summary.csv"
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

    frames = [bm25, vector]

    # Graph RAG (Ziyad): "Graph" and "Graph+BM25" retrievers
    if GRAPH_FILE.exists():
        graph = pd.read_csv(GRAPH_FILE)
        graph = graph[graph["k"] == 10].copy()
        frames.append(graph[columns + ["retriever"]])

    comparison = pd.concat(
        frames,
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

    print("\nBM25 VS VECTOR VS GRAPH — K=10")
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