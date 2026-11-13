import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_FILE = Path("results/bm25_summary.csv")
OUTPUT_DIR = Path("results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_FILE)

    # Separate hop-specific results from overall
    hop_df = df[df["hop_count"] != "Overall"].copy()
    overall_df = df[df["hop_count"] == "Overall"].copy()

    hop_df["hop_count"] = hop_df["hop_count"].astype(int)

    return hop_df, overall_df


def plot_recall_by_k(hop_df):
    plt.figure(figsize=(8, 5))

    for hop in [2, 3, 4, 5]:
        subset = hop_df[hop_df["hop_count"] == hop]

        plt.plot(
            subset["k"],
            subset["recall"],
            marker="o",
            label=f"{hop}-hop"
        )

    plt.xlabel("Top-K Retrieved Chunks")
    plt.ylabel("Recall@K")
    plt.title("BM25 Recall by Reasoning Depth")
    plt.xticks([1, 3, 5, 10])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "bm25_recall_by_hop.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def plot_precision_by_k(hop_df):
    plt.figure(figsize=(8, 5))

    for hop in [2, 3, 4, 5]:
        subset = hop_df[hop_df["hop_count"] == hop]

        plt.plot(
            subset["k"],
            subset["precision"],
            marker="o",
            label=f"{hop}-hop"
        )

    plt.xlabel("Top-K Retrieved Chunks")
    plt.ylabel("Precision@K")
    plt.title("BM25 Precision by Reasoning Depth")
    plt.xticks([1, 3, 5, 10])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "bm25_precision_by_hop.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def plot_evidence_recovery(hop_df):
    plt.figure(figsize=(8, 5))

    for hop in [2, 3, 4, 5]:
        subset = hop_df[hop_df["hop_count"] == hop]

        plt.plot(
            subset["k"],
            subset["evidence_recovery"],
            marker="o",
            label=f"{hop}-hop"
        )

    plt.xlabel("Top-K Retrieved Chunks")
    plt.ylabel("Full Evidence Recovery Rate")
    plt.title("BM25 Full Evidence Recovery by Reasoning Depth")
    plt.xticks([1, 3, 5, 10])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "bm25_evidence_recovery.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def plot_overall(overall_df):
    overall_df = overall_df.sort_values("k")

    plt.figure(figsize=(8, 5))

    plt.plot(
        overall_df["k"],
        overall_df["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        overall_df["k"],
        overall_df["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        overall_df["k"],
        overall_df["evidence_recovery"],
        marker="o",
        label="Full Evidence Recovery"
    )

    plt.xlabel("Top-K Retrieved Chunks")
    plt.ylabel("Score")
    plt.title("Overall BM25 Retrieval Performance")
    plt.xticks([1, 3, 5, 10])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "bm25_overall_performance.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def main():
    hop_df, overall_df = load_data()

    plot_recall_by_k(hop_df)
    plot_precision_by_k(hop_df)
    plot_evidence_recovery(hop_df)
    plot_overall(overall_df)

    print("\nAll BM25 plots generated.")


if __name__ == "__main__":
    main()