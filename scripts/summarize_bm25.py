import pandas as pd
from pathlib import Path


INPUT_FILE = Path("results/bm25_results.csv")
OUTPUT_FILE = Path("results/bm25_summary.csv")


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} result rows")
    print(f"Unique questions: {df['question_id'].nunique()}")

    # ---------------------------------------------------------
    # Results by hop count and K
    # ---------------------------------------------------------

    summary = (
        df.groupby(["hop_count", "k"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            mrr=("reciprocal_rank", "mean"),
            evidence_recovery=("evidence_recovery", "mean"),
            latency_ms=("retrieval_latency_ms", "mean"),
            questions=("question_id", "nunique")
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Overall results
    # ---------------------------------------------------------

    overall = (
        df.groupby("k")
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            mrr=("reciprocal_rank", "mean"),
            evidence_recovery=("evidence_recovery", "mean"),
            latency_ms=("retrieval_latency_ms", "mean"),
            questions=("question_id", "nunique")
        )
        .reset_index()
    )

    overall["hop_count"] = "Overall"

    # Reorder
    overall = overall[
        [
            "hop_count",
            "k",
            "precision",
            "recall",
            "mrr",
            "evidence_recovery",
            "latency_ms",
            "questions"
        ]
    ]

    final = pd.concat(
        [summary, overall],
        ignore_index=True
    )

    # Save full summary
    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Print readable results
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("BM25 RESULTS BY REASONING DEPTH")
    print("=" * 90)

    for hop in [2, 3, 4, 5]:

        print(f"\n{hop}-HOP")

        subset = summary[
            summary["hop_count"] == hop
        ]

        for _, row in subset.iterrows():

            print(
                f"K={int(row['k']):2d} | "
                f"P@K={row['precision']:.3f} | "
                f"R@K={row['recall']:.3f} | "
                f"MRR={row['mrr']:.3f} | "
                f"Full Evidence={row['evidence_recovery']:.3f}"
            )

    print("\nOVERALL")

    for _, row in overall.iterrows():

        print(
            f"K={int(row['k']):2d} | "
            f"P@K={row['precision']:.3f} | "
            f"R@K={row['recall']:.3f} | "
            f"MRR={row['mrr']:.3f} | "
            f"Full Evidence={row['evidence_recovery']:.3f}"
        )

    print("\n" + "=" * 90)

    avg_latency = df[
        df["k"] == df["k"].max()
    ]["retrieval_latency_ms"].mean()

    median_latency = df[
        df["k"] == df["k"].max()
    ]["retrieval_latency_ms"].median()

    print(
        f"Average retrieval latency: "
        f"{avg_latency:.3f} ms/query"
    )

    print(
        f"Median retrieval latency: "
        f"{median_latency:.3f} ms/query"
    )

    print(f"\nSaved summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()