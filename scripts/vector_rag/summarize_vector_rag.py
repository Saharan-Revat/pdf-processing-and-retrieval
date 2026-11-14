import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "results/vector/vector_rag_results.csv"
)

OUTPUT_FILE = Path(
    "results/vector/vector_rag_summary.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Questions: {len(df)}")

    summary = (
        df.groupby("hop_count")
        .agg(
            questions=("question_id", "count"),

            correct_answers=(
                "answer_match",
                "sum"
            ),

            answer_accuracy=(
                "answer_match",
                "mean"
            ),

            any_evidence_rate=(
                "any_evidence_retrieved",
                "mean"
            ),

            full_evidence_rate=(
                "full_evidence_retrieved",
                "mean"
            ),

            avg_retrieval_latency_ms=(
                "retrieval_latency_ms",
                "mean"
            ),

            avg_generation_latency_ms=(
                "generation_latency_ms",
                "mean"
            )
        )
        .reset_index()
    )

    overall = pd.DataFrame([{
        "hop_count": "Overall",

        "questions":
            len(df),

        "correct_answers":
            df["answer_match"].sum(),

        "answer_accuracy":
            df["answer_match"].mean(),

        "any_evidence_rate":
            df[
                "any_evidence_retrieved"
            ].mean(),

        "full_evidence_rate":
            df[
                "full_evidence_retrieved"
            ].mean(),

        "avg_retrieval_latency_ms":
            df[
                "retrieval_latency_ms"
            ].mean(),

        "avg_generation_latency_ms":
            df[
                "generation_latency_ms"
            ].mean()
    }])

    final = pd.concat(
        [summary, overall],
        ignore_index=True
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 100)
    print("VECTOR RAG RESULTS BY REASONING DEPTH")
    print("=" * 100)

    for _, row in final.iterrows():

        hop = row["hop_count"]

        print(
            f"{str(hop):>7} | "
            f"Questions={int(row['questions']):2d} | "
            f"Correct={int(row['correct_answers']):2d} | "
            f"Accuracy={row['answer_accuracy']:.1%} | "
            f"Any Evidence={row['any_evidence_rate']:.1%} | "
            f"Full Evidence={row['full_evidence_rate']:.1%}"
        )

    # ---------------------------------------------------------
    # Conditional generation accuracy
    # ---------------------------------------------------------

    evidence_df = df[
        df["any_evidence_retrieved"] == True
    ]

    full_df = df[
        df["full_evidence_retrieved"] == True
    ]

    print("\n" + "=" * 100)
    print("GENERATION CONDITIONAL ON RETRIEVAL")
    print("=" * 100)

    if len(evidence_df) > 0:

        print(
            "Accuracy when ANY gold evidence was retrieved: "
            f"{evidence_df['answer_match'].mean():.1%} "
            f"({int(evidence_df['answer_match'].sum())}/"
            f"{len(evidence_df)})"
        )

    if len(full_df) > 0:

        print(
            "Accuracy when FULL gold evidence was retrieved: "
            f"{full_df['answer_match'].mean():.1%} "
            f"({int(full_df['answer_match'].sum())}/"
            f"{len(full_df)})"
        )

    print(
        f"\nSaved summary to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()