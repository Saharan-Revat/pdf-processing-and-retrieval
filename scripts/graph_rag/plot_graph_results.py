"""
Plots for the report.

    python scripts/graph_rag/plot_graph_results.py

  results/graph/plots/retriever_<metric>_by_hop.png  BM25 vs Vector vs Graph vs Graph+BM25
  results/graph/plots/kg_<contract>.png               knowledge-graph excerpt of one contract
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from kg_common import GRAPH_FILE, RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"

SUMMARIES = {
    "BM25": Path("results/bm25_summary.csv"),
    "Vector": Path("results/vector/vector_summary.csv"),
}

COLORS = {"BM25": "#4C72B0", "Vector": "#DD8452", "Graph": "#55A868", "Graph+BM25": "#8172B3"}

METRICS = {
    "recall": "Recall@10",
    "evidence_recovery": "Full Evidence Recovery@10",
    "precision": "Precision@10",
    "mrr": "MRR",
}


def load_all(k=10):

    frames = []

    for name, path in SUMMARIES.items():
        if path.exists():
            df = pd.read_csv(path)
            df["retriever"] = name
            frames.append(df)

    graph = pd.read_csv(RESULTS_DIR / "graph_summary.csv")
    frames.append(graph)

    df = pd.concat(frames, ignore_index=True)
    df["hop_count"] = df["hop_count"].astype(str)
    return df[df["k"] == k]


def plot_metric(df, metric, label):

    hops = ["2", "3", "4", "5", "Overall"]
    retrievers = [r for r in COLORS if r in set(df["retriever"])]
    width = 0.8 / len(retrievers)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, retriever in enumerate(retrievers):
        sub = df[df["retriever"] == retriever].set_index("hop_count")
        values = [sub.loc[h, metric] if h in sub.index else 0 for h in hops]
        positions = [x + (i - (len(retrievers) - 1) / 2) * width for x in range(len(hops))]
        bars = ax.bar(positions, values, width, label=retriever, color=COLORS[retriever])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(range(len(hops)))
    ax.set_xticklabels([f"{h}-hop" if h != "Overall" else h for h in hops])
    ax.set_ylabel(label)
    ax.set_ylim(0, min(1.0, df[metric].max() * 1.25 + 0.02))
    ax.set_title(f"{label} by reasoning depth")
    ax.legend(frameon=False, ncol=len(retrievers), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / f"retriever_{metric}_by_hop.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def plot_knowledge_graph(document_prefix="DYNAMEX", max_entities=30):

    data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))

    nodes = {n["id"]: n for n in data["nodes"] if n["document_id"].startswith(document_prefix)}
    if not nodes:
        return

    weak = {"MENTIONED_IN", "STATED_IN", "HAS_CHUNK", "NEXT",
            "CONTAINS_SECTION", "REFERS_TO", "PART_OF", "DEFINED_IN"}

    graph = nx.Graph()
    for e in data["edges"]:
        if e["source"] in nodes and e["target"] in nodes and e["relation"] not in weak:
            a, b = nodes[e["source"]], nodes[e["target"]]
            if a["label"] == "Entity" and b["label"] in ("Entity", "Statement"):
                graph.add_edge(e["source"], e["target"], relation=e["relation"])

    degree = sorted(((n, d) for n, d in graph.degree() if nodes[n]["label"] == "Entity"),
                    key=lambda x: -x[1])
    keep_entities = {n for n, _ in degree[:max_entities]}
    keep = set(keep_entities)
    for n in keep_entities:
        statements = [m for m in graph.neighbors(n) if nodes[m]["label"] == "Statement"
                      and len(nodes[m]["name"].split()) > 3][:2]
        keep.update(statements)
    sub = graph.subgraph(keep).copy()

    palette = {"Party": "#C44E52", "Organization": "#C44E52", "DefinedTerm": "#4C72B0",
               "Duration": "#55A868", "Amount": "#55A868", "Percentage": "#55A868",
               "Date": "#55A868", "Location": "#8172B3"}

    colors, sizes, labels = [], [], {}
    for n in sub.nodes:
        info = nodes[n]
        if info["label"] == "Statement":
            colors.append("#CCCCCC")
            sizes.append(120)
            labels[n] = textwrap.shorten(info["name"], 38, placeholder="…")
        else:
            colors.append(palette.get(info.get("entity_type"), "#937860"))
            sizes.append(300 + 80 * sub.degree(n))
            labels[n] = info["name"][:28]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(sub, k=0.9, seed=7, iterations=200)
    strong = [(u, v) for u, v, d in sub.edges(data=True) if d["relation"] != "CO_OCCURS_WITH"]
    co = [(u, v) for u, v, d in sub.edges(data=True) if d["relation"] == "CO_OCCURS_WITH"]
    nx.draw_networkx_edges(sub, pos, edgelist=co, ax=ax, alpha=0.15, width=0.6, style="dashed")
    nx.draw_networkx_edges(sub, pos, edgelist=strong, ax=ax, alpha=0.5, width=1.0)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_labels(sub, pos, labels, ax=ax, font_size=7)
    edge_labels = {(u, v): d["relation"] for u, v, d in sub.edges(data=True)
                   if d["relation"] != "CO_OCCURS_WITH"}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels, ax=ax, font_size=5.5, alpha=0.8)

    document = next(iter(nodes.values()))["document_id"]
    ax.set_title(f"Knowledge graph excerpt: {document[:80]}\n"
                 "red = party/organisation, blue = defined term, green = value, grey = obligation/definition, dashed = co-occurrence",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    path = PLOTS_DIR / f"kg_{document_prefix.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    for metric, label in METRICS.items():
        plot_metric(df, metric, label)
    plot_knowledge_graph(sys.argv[1] if len(sys.argv) > 1 else "DYNAMEX")


if __name__ == "__main__":
    main()
