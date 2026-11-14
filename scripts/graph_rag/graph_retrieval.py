"""
Graph retrieval over the contract knowledge graph.

Pipeline (per question, restricted to the question's contract like the
BM25 and Vector baselines):

  1. Entity matching   - question text is matched against entity names and
                         aliases (exact phrase, then partial token match) and
                         against section numbers ("sections 3.1 and 3.2").
                         Each seed is weighted by IDF over chunks, so rare,
                         specific entities ("Royalty Rate") dominate generic
                         ones ("Product").
  2. Graph traversal   - Personalised PageRank (random walk with restart)
                         from the seeds over the document graph. The restart
                         probability (1 - ALPHA) bounds the walk to a local
                         neighbourhood (expected path length ~ 1/(1-ALPHA)).
                         Edges: entity-chunk mentions, definitions,
                         entity-entity relations, chunk->section references,
                         section containment and chunk adjacency.
  3. Evidence ranking  - Chunk nodes are ranked by their walk score (with a
                         mild degree normalisation against hub chunks).
  4. Provenance        - for every returned chunk the shortest path from a
                         seed node is reported, e.g.
                         'Royalty Rate' -DEFINED_IN-> chunk 12.

Two modes are exposed:

  graph   pure Graph RAG: seeds come only from entity/section matching.
  hybrid  graph-enhanced retrieval: BM25's top chunks are added as extra
          seeds, then the graph expands to definitions / cross-referenced
          sections that lexical matching alone misses.
"""

import json
import math
import re
from collections import defaultdict

import networkx as nx

from kg_common import (
    CHUNKS_DIR,
    GRAPH_FILE,
    build_bm25_indexes,
    get_question_document,
    load_chunks,
    normalize_question,
    retrieve_bm25,
)

ALPHA = 0.75                # PageRank damping = 1 - restart probability
DEGREE_NORMALISATION = 0.5  # chunk score / degree ** beta
HYBRID_BM25_SEEDS = 5
HYBRID_BM25_WEIGHT = 0.5    # share of restart mass given to BM25 seeds

EDGE_WEIGHTS = {
    "MENTIONED_IN": 1.0,
    "DEFINED_IN": 3.0,
    "STATED_IN": 1.0,
    "CONTAINS_SECTION": 2.0,
    "REFERS_TO": 1.5,
    "PART_OF": 0.5,
    "NEXT": 0.3,
    "CO_OCCURS_WITH": 0.3,
    "ALSO_KNOWN_AS": 3.0,
    "HAS_CHUNK": 0.0,       # document node is not used for retrieval
}
DEFAULT_RELATION_WEIGHT = 1.0

STOP_KEYS = {
    "agreement", "party", "parties", "company", "contract", "term",
    "section", "date", "product", "products", "services", "service",
}

SECTION_IN_QUESTION = re.compile(
    r"\b(?:sections?|articles?|paragraphs?|clauses?|subsections?)\s+"
    r"((?:\d{1,2}(?:\.\d{1,2}){0,3}(?:\s*\([a-z0-9]{1,4}\))*(?:\s*(?:,|and|or|through)\s*)?)+)",
    re.I,
)


class DocumentGraph:

    def __init__(self, document_id):
        self.document_id = document_id
        self.graph = nx.Graph()
        self.relations = {}             # (u, v) -> relation label for paths
        self.entity_keys = defaultdict(set)   # match key -> {entity node ids}
        self.sections = {}              # "3.1" -> node id
        self.chunk_nodes = []
        self.node_names = {}
        self.chunk_df = {}              # entity node -> number of chunks

    def add_edge(self, source, relation, target, count):
        weight = EDGE_WEIGHTS.get(relation, DEFAULT_RELATION_WEIGHT)
        if weight <= 0:
            return
        weight *= 1.0 + math.log(count) if count > 1 else 1.0
        if self.graph.has_edge(source, target):
            self.graph[source][target]["weight"] += weight
        else:
            self.graph.add_edge(source, target, weight=weight)
            self.relations[(source, target)] = relation


def load_graph_system(graph_file=GRAPH_FILE):

    data = json.loads(graph_file.read_text(encoding="utf-8"))

    documents = {}
    node_document = {}

    for node in data["nodes"]:
        doc = documents.setdefault(node["document_id"], DocumentGraph(node["document_id"]))
        node_document[node["id"]] = node["document_id"]

        if node["label"] == "Document":
            continue

        doc.graph.add_node(node["id"], label=node["label"])
        doc.node_names[node["id"]] = node.get("name", node["id"])

        if node["label"] == "Chunk":
            doc.chunk_nodes.append(node["id"])
        elif node["label"] == "Entity":
            for key in node.get("match_keys", [node.get("key", "")]):
                if key and len(key) >= 2:
                    doc.entity_keys[key].add(node["id"])
        elif node["label"] == "Section":
            doc.sections[node["number"]] = node["id"]

    for edge in data["edges"]:
        doc = documents[node_document[edge["source"]]]
        if edge["source"] in doc.graph and edge["target"] in doc.graph:
            doc.add_edge(edge["source"], edge["relation"], edge["target"], edge.get("count", 1))

    for doc in documents.values():
        chunk_set = set(doc.chunk_nodes)
        for keys in doc.entity_keys.values():
            for node_id in keys:
                doc.chunk_df[node_id] = sum(1 for n in doc.graph.neighbors(node_id) if n in chunk_set)

    return {"documents": documents, "extractor": data.get("extractor")}


# ---------------------------------------------------------------
# Step 1: entity / section matching
# ---------------------------------------------------------------

def match_seeds(question_text, doc):

    q = normalize_question(question_text)
    q_tokens = set(q.split())
    n_chunks = max(1, len(doc.chunk_nodes))

    seeds = defaultdict(float)
    matched = []

    def idf(node_id):
        df = doc.chunk_df.get(node_id, 0)
        return math.log(1.0 + n_chunks / (1.0 + df))

    for key, node_ids in doc.entity_keys.items():

        if key in STOP_KEYS:
            continue

        exact = f" {key} " in q
        partial = False

        if not exact:
            tokens = [t for t in re.findall(r"\w+", key) if len(t) > 3]
            if len(tokens) >= 2 and all(t in q_tokens for t in tokens):
                partial = True

        if not (exact or partial):
            continue

        strength = 1.0 if exact else 0.5
        # longer phrases are more specific
        strength *= 1.0 + 0.25 * min(3, len(key.split()) - 1)

        for node_id in node_ids:
            weight = strength * idf(node_id)
            if weight > seeds[node_id]:
                seeds[node_id] = weight
                matched.append(doc.node_names[node_id])

    for match in SECTION_IN_QUESTION.finditer(question_text):
        for number in re.findall(r"\d{1,2}(?:\.\d{1,2}){0,3}", match.group(1)):
            number = re.sub(r"\.0$", "", number)
            node_id = doc.sections.get(number)
            if node_id:
                seeds[node_id] = max(seeds[node_id], 2.0)
                matched.append(f"Section {number}")

    # Remove seeds subsumed by a longer matched phrase
    # ("Product" when "Licensed Product" also matched)
    names = {n: normalize_question(doc.node_names[n]).strip() for n in seeds}
    for a in list(seeds):
        for b in seeds:
            if a != b and names[a] and names[a] != names[b] and f" {names[a]} " in f" {names[b]} ":
                seeds[a] *= 0.3
                break

    return dict(seeds), sorted(set(matched))


def fallback_seeds(question_text, doc):
    """
    No entity matched: seed entities sharing content words with the question.
    """

    q_tokens = {t for t in normalize_question(question_text).split() if len(t) > 3}
    seeds = {}

    for key, node_ids in doc.entity_keys.items():
        tokens = {t for t in re.findall(r"\w+", key) if len(t) > 3}
        overlap = len(tokens & q_tokens)
        if overlap:
            for node_id in node_ids:
                seeds[node_id] = overlap / len(tokens)

    return seeds


# ---------------------------------------------------------------
# Steps 2-4: traversal, ranking, provenance
# ---------------------------------------------------------------

def personalised_pagerank(graph, seeds):
    total = sum(seeds.values())
    personalization = {n: w / total for n, w in seeds.items()}
    return nx.pagerank(graph, alpha=ALPHA, personalization=personalization,
                       weight="weight", max_iter=200, tol=1e-08)


def evidence_path(doc, seeds, target, max_hops=4):

    best = None

    for seed in sorted(seeds, key=seeds.get, reverse=True)[:8]:
        try:
            path = nx.shortest_path(doc.graph, seed, target)
        except nx.NetworkXNoPath:
            continue
        if len(path) - 1 <= max_hops and (best is None or len(path) < len(best)):
            best = path

    if not best:
        return ""

    def display(node_id):
        name = doc.node_names[node_id]
        if doc.graph.nodes[node_id]["label"] == "Chunk":
            return "chunk " + name.rsplit("::", 1)[-1]
        return f"'{name[:60]}'"

    parts = [display(best[0]) + (" [BM25 seed]" if doc.graph.nodes[best[0]]["label"] == "Chunk" else "")]
    for u, v in zip(best, best[1:]):
        relation = doc.relations.get((u, v)) or doc.relations.get((v, u)) or "RELATED"
        parts.append(f"-{relation}- {display(v)}")

    return " ".join(parts)


def retrieve_graph(question, system, chunk_lookup, top_k=10, mode="graph",
                   bm25_indexes=None, with_paths=True):

    document_id = get_question_document(question)
    doc = system["documents"].get(document_id)

    if doc is None:
        raise ValueError(f"Document not in knowledge graph: {document_id}")

    seeds, matched = match_seeds(question["question"], doc)
    seed_source = "entities"

    if not seeds:
        seeds = fallback_seeds(question["question"], doc)
        seed_source = "fallback"

    if mode == "hybrid":
        if bm25_indexes is None:
            raise ValueError("hybrid mode needs bm25_indexes")
        bm25_hits = retrieve_bm25(question, bm25_indexes, top_k=HYBRID_BM25_SEEDS)
        entity_mass = sum(seeds.values()) or 1.0
        bm25_total = sum(max(h["score"], 0.0) for h in bm25_hits) or 1.0
        share = HYBRID_BM25_WEIGHT if seeds else 1.0
        scale = entity_mass * share / (1.0 - share) if seeds and share < 1 else 1.0
        combined = dict(seeds)
        for hit in bm25_hits:
            node_id = f"C|{hit['chunk_id']}"
            combined[node_id] = combined.get(node_id, 0.0) + scale * max(hit["score"], 1e-6) / bm25_total
        seeds = combined
        seed_source += "+bm25"

    if not seeds:
        seeds = {n: 1.0 for n in doc.chunk_nodes}
        seed_source = "uniform"

    scores = personalised_pagerank(doc.graph, seeds)

    ranked = sorted(
        doc.chunk_nodes,
        key=lambda n: scores.get(n, 0.0) / (max(1, doc.graph.degree(n)) ** DEGREE_NORMALISATION),
        reverse=True,
    )

    results = []
    for rank, node_id in enumerate(ranked[:top_k], start=1):
        chunk_id = node_id[2:]
        chunk = chunk_lookup[chunk_id]
        results.append({
            "rank": rank,
            "score": float(scores.get(node_id, 0.0)),
            "chunk_id": chunk_id,
            "document_id": document_id,
            "pages": chunk.get("page_numbers", []),
            "text": chunk["text"],
            "evidence_path": evidence_path(doc, seeds, node_id) if with_paths and seed_source != "uniform" else "",
        })

    return results, {"matched_entities": matched, "seed_source": seed_source, "num_seeds": len(seeds)}


def load_chunk_lookup():
    return {c["chunk_id"]: c for c in load_chunks(CHUNKS_DIR)}


if __name__ == "__main__":

    from kg_common import EVALUATION_DIR, load_evaluation

    system = load_graph_system()
    lookup = load_chunk_lookup()
    indexes = build_bm25_indexes(list(lookup.values()))
    questions = load_evaluation(EVALUATION_DIR)

    for question in questions[:3]:
        print("\n" + "=" * 80)
        print(question["question"])
        print("Gold:", question["gold_answer"])
        for mode in ("graph", "hybrid"):
            results, info = retrieve_graph(question, system, lookup, top_k=5, mode=mode, bm25_indexes=indexes)
            print(f"\n[{mode}] matched={info['matched_entities']} seeds={info['seed_source']}")
            for r in results:
                print(f"  {r['rank']}. {r['chunk_id'].rsplit('::', 1)[-1]:>3}  {r['score']:.4f}  {r['evidence_path'][:110]}")
