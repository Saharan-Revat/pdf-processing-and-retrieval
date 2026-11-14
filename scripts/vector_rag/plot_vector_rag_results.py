import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_FILE = Path(
    "results/vector/vector_rag_summary.csv"
)

OUTPUT_DIR = Path(
    "results/vector/plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    df = pd.read_csv(INPUT_FILE)

    # Remove overall row
    df = df[
        df["hop_count"] != "Overall"
    ].copy()

    df["hop_count"] = (
        df["hop_count"].astype(int)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["hop_count"],
        df["answer_accuracy"],
        marker="o",
        label="Gold Answer Match"
    )

    plt.plot(
        df["hop_count"],
        df["any_evidence_rate"],
        marker="o",
        label="Any Gold Evidence"
    )

    plt.plot(
        df["hop_count"],
        df["full_evidence_rate"],
        marker="o",
        label="Full Gold Evidence"
    )

    plt.xlabel("Reasoning Depth (Hop Count)")
    plt.ylabel("Rate")

    plt.title(
        "Vector RAG Performance by Reasoning Depth"
    )

    plt.xticks([2, 3, 4, 5])
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "vector_rag_by_hop.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved {output_path}"
    )


if __name__ == "__main__":
    main()