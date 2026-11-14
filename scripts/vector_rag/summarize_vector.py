import pandas as pd
from pathlib import Path


INPUT_FILE = Path("results/vector/vector_results.csv")
OUTPUT_FILE = Path("results/vector/vector_summary.csv")


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} result rows")
    print(
        f"Unique questions: "
        f"{df['question_id'].nunique()}"
    )

    # ---------------------------------------------------------
    # Results by hop and K
    # ---------------------------------------------------------

    summary = (
        df.groupby(["hop_count", "k"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            mrr=("reciprocal_rank", "mean"),
            evidence_recovery=(
                "evidence_recovery",
                "mean"
            ),
            latency_ms=(
                "retrieval_latency_ms",
                "mean"
            ),
            questions=(
                "question_id",
                "nunique"
            )
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Overall
    # ---------------------------------------------------------

    overall = (
        df.groupby("k")
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            mrr=("reciprocal_rank", "mean"),
            evidence_recovery=(
                "evidence_recovery",
                "mean"
            ),
            latency_ms=(
                "retrieval_latency_ms",
                "mean"
            ),
            questions=(
                "question_id",
                "nunique"
            )
        )
        .reset_index()
    )

    overall["hop_count"] = "Overall"

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

    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("VECTOR RESULTS BY REASONING DEPTH")
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
                f"Full Evidence="
                f"{row['evidence_recovery']:.3f}"
            )

    print("\nOVERALL")

    for _, row in overall.iterrows():

        print(
            f"K={int(row['k']):2d} | "
            f"P@K={row['precision']:.3f} | "
            f"R@K={row['recall']:.3f} | "
            f"MRR={row['mrr']:.3f} | "
            f"Full Evidence="
            f"{row['evidence_recovery']:.3f}"
        )

    print("\n" + "=" * 90)

    latency_df = df[
        df["k"] == df["k"].max()
    ]

    print(
        "Average retrieval latency: "
        f"{latency_df['retrieval_latency_ms'].mean():.3f} "
        "ms/query"
    )

    print(
        "Median retrieval latency: "
        f"{latency_df['retrieval_latency_ms'].median():.3f} "
        "ms/query"
    )

    print(
        f"\nSaved summary to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()