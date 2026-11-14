"""
Build the knowledge graph from extracted triples.

    python scripts/graph_rag/build_graph.py

Input : data/knowledge_graph/triples/*.json   (kg_extraction.py)
Output: data/knowledge_graph/graph.json       nodes + typed edges w/ provenance
        data/knowledge_graph/graph_stats.json
        data/knowledge_graph/graph.graphml    (Gephi / yEd / Cytoscape)
        data/knowledge_graph/neo4j_import/*.csv

Graph schema (one connected sub-graph per contract):

  (:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT]->(:Chunk)
  (:Entity)-[:MENTIONED_IN]->(:Chunk)
  (:Entity)-[:DEFINED_IN]->(:Chunk)
  (:Entity)-[:<PREDICATE> {chunk_ids}]->(:Entity)      e.g. ALSO_KNOWN_AS,
                                                         HAS_DURATION, MUST ...
  (:Entity)-[:<PREDICATE>]->(:Statement)                obligations, definitions
  (:Statement)-[:STATED_IN]->(:Chunk)
  (:Chunk)-[:CONTAINS_SECTION]->(:Section)
  (:Chunk)-[:REFERS_TO]->(:Section)
  (:Section)-[:PART_OF]->(:Section)                     3.1 -> 3

Normalisation:
  * entity names are canonicalised (case, quotes, articles, possessives,
    corporate suffixes) by kg_common.normalize_entity;
  * ALSO_KNOWN_AS triples are merged with union-find, so
    "PUROLATOR COURIER LTD." and "Purolator" become one node;
  * the most frequent surface form becomes the display name, the others are
    kept as aliases.
"""

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict

from kg_common import (
    GRAPH_FILE,
    GRAPH_STATS_FILE,
    KG_DIR,
    NEO4J_EXPORT_DIR,
    TRIPLES_DIR,
    document_slug,
    normalize_entity,
)

STATEMENT_TYPES = {"Obligation", "Definition", "Right"}


class UnionFind:

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the shorter key as the canonical one ("purolator")
            if (len(ra), ra) <= (len(rb), rb):
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def short_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def section_parents(number):
    parts = number.split(".")
    return [".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def build_document_graph(doc):

    document_id = doc["document_id"]
    slug = document_slug(document_id)
    records = doc["chunks"]

    nodes = {}
    edges = defaultdict(lambda: {"chunk_ids": set(), "count": 0})

    def node(node_id, **attrs):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "document_id": document_id, **attrs}
        return node_id

    def edge(src, relation, target, chunk_id=None, **attrs):
        key = (src, relation, target)
        data = edges[key]
        data["count"] += 1
        if chunk_id:
            data["chunk_ids"].add(chunk_id)
        for name, value in attrs.items():
            data.setdefault(name, value)

    # ---------- alias resolution ---------------------------------

    uf = UnionFind()
    for record in records:
        for triple in record["triples"]:
            if triple["predicate"] == "ALSO_KNOWN_AS":
                uf.union(normalize_entity(triple["subject"]), normalize_entity(triple["object"]))

    surfaces = defaultdict(Counter)
    types = defaultdict(Counter)

    def entity_id(name, kind):
        key = uf.find(normalize_entity(name))
        if not key:
            return None
        surfaces[key][name.strip()] += 1
        types[key][kind] += 1
        return f"E|{slug}|{key}"

    # ---------- document & chunk nodes ---------------------------

    doc_node = node(f"D|{slug}", label="Document", name=document_id)
    ordered = sorted(records, key=lambda r: int(r["chunk_id"].rsplit("::", 1)[-1]))

    for index, record in enumerate(ordered):
        chunk_node = node(
            f"C|{record['chunk_id']}", label="Chunk", name=record["chunk_id"],
            chunk_id=record["chunk_id"], chunk_index=index,
            pages=record.get("page_numbers", []),
        )
        edge(doc_node, "HAS_CHUNK", chunk_node)
        if index:
            edge(f"C|{ordered[index - 1]['chunk_id']}", "NEXT", chunk_node)

    # ---------- entities, relations, sections --------------------

    for record in records:

        chunk_id = record["chunk_id"]
        chunk_node = f"C|{chunk_id}"

        for entity in record["entities"]:
            eid = entity_id(entity["name"], entity["type"])
            if eid:
                node(eid, label="Entity")
                edge(eid, "MENTIONED_IN", chunk_node, chunk_id)

        for triple in record["triples"]:

            sid = entity_id(triple["subject"], triple["subject_type"])
            if not sid:
                continue
            node(sid, label="Entity")
            edge(sid, "MENTIONED_IN", chunk_node, chunk_id)

            predicate = triple["predicate"]

            if triple["object_type"] in STATEMENT_TYPES:
                text = triple["object"]
                st_id = f"T|{slug}|{short_hash(sid + predicate + text)}"
                node(st_id, label="Statement", name=text[:200],
                     statement_type=triple["object_type"])
                edge(sid, predicate, st_id, chunk_id, extractor=triple.get("source", "rules"))
                edge(st_id, "STATED_IN", chunk_node, chunk_id)
                if predicate == "DEFINED_AS":
                    edge(sid, "DEFINED_IN", chunk_node, chunk_id)
                continue

            oid = entity_id(triple["object"], triple["object_type"])
            if not oid or oid == sid:
                continue
            node(oid, label="Entity")
            edge(oid, "MENTIONED_IN", chunk_node, chunk_id)
            edge(sid, predicate, oid, chunk_id, extractor=triple.get("source", "rules"),
                 evidence=triple.get("evidence", "")[:300])

        titles = record.get("section_titles", {})

        for number in record.get("sections_defined", []):
            sec = node(f"S|{slug}|{number}", label="Section", name=f"Section {number}",
                       number=number)
            if titles.get(number):
                nodes[sec].setdefault("title", titles[number])
            edge(chunk_node, "CONTAINS_SECTION", sec, chunk_id)

        for number in record.get("sections_referenced", []):
            sec = node(f"S|{slug}|{number}", label="Section", name=f"Section {number}",
                       number=number)
            edge(chunk_node, "REFERS_TO", sec, chunk_id)

    # section hierarchy
    for node_id in list(nodes):
        if nodes[node_id]["label"] == "Section":
            for parent in section_parents(nodes[node_id]["number"]):
                parent_id = f"S|{slug}|{parent}"
                if parent_id in nodes:
                    edge(node_id, "PART_OF", parent_id)
                    break

    # ---------- finalise entity attributes -----------------------

    for node_id, data in nodes.items():
        if data["label"] != "Entity":
            continue
        key = node_id.split("|", 2)[2]
        data["name"] = surfaces[key].most_common(1)[0][0] if surfaces[key] else key
        data["key"] = key
        data["entity_type"] = types[key].most_common(1)[0][0] if types[key] else "Other"
        data["aliases"] = sorted({s for s in surfaces[key]} - {data["name"]})[:10]
        data["match_keys"] = sorted({normalize_entity(s) for s in surfaces[key]} | {key})

    edge_list = [
        {
            "source": s, "relation": r, "target": t,
            "count": d["count"], "chunk_ids": sorted(d["chunk_ids"]),
            **{k: v for k, v in d.items() if k not in ("chunk_ids", "count")},
        }
        for (s, r, t), d in edges.items()
    ]

    return list(nodes.values()), edge_list


def export_neo4j_csv(nodes, edges):

    NEO4J_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(NEO4J_EXPORT_DIR / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id:ID", ":LABEL", "name", "document_id", "entity_type",
                         "chunk_id", "number", "title", "aliases"])
        for n in nodes:
            labels = [n["label"]]
            if n["label"] == "Entity":
                labels.append(n.get("entity_type", "Other").replace(" ", ""))
            writer.writerow([n["id"], ";".join(labels), n.get("name", ""), n["document_id"],
                             n.get("entity_type", ""), n.get("chunk_id", ""),
                             n.get("number", ""), n.get("title", ""),
                             "|".join(n.get("aliases", []))])

    with open(NEO4J_EXPORT_DIR / "relationships.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID", ":END_ID", ":TYPE", "count:int", "chunk_ids", "extractor", "evidence"])
        for e in edges:
            writer.writerow([e["source"], e["target"], e["relation"], e["count"],
                             "|".join(e["chunk_ids"]), e.get("extractor", ""),
                             e.get("evidence", "")])


def export_graphml(nodes, edges):
    import networkx as nx

    graph = nx.MultiDiGraph()
    for n in nodes:
        graph.add_node(n["id"], label=n["label"], name=str(n.get("name", ""))[:200],
                       entity_type=n.get("entity_type", ""), document_id=n["document_id"])
    for e in edges:
        graph.add_edge(e["source"], e["target"], relation=e["relation"], count=e["count"])
    nx.write_graphml(graph, KG_DIR / "graph.graphml")


def main():

    start = time.perf_counter()

    files = sorted(TRIPLES_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No triples found. Run kg_extraction.py first.")

    all_nodes, all_edges, per_document = [], [], {}
    extractor = None

    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        extractor = doc.get("extractor")
        nodes, edges = build_document_graph(doc)
        all_nodes += nodes
        all_edges += edges
        per_document[doc["document_id"]] = {
            "nodes": len(nodes),
            "edges": len(edges),
            "entities": sum(1 for n in nodes if n["label"] == "Entity"),
            "sections": sum(1 for n in nodes if n["label"] == "Section"),
        }

    build_time = time.perf_counter() - start

    GRAPH_FILE.write_text(json.dumps({"extractor": extractor, "nodes": all_nodes, "edges": all_edges},
                                     ensure_ascii=False), encoding="utf-8")

    export_neo4j_csv(all_nodes, all_edges)
    export_graphml(all_nodes, all_edges)

    node_labels = Counter(n["label"] for n in all_nodes)
    entity_types = Counter(n.get("entity_type") for n in all_nodes if n["label"] == "Entity")
    relations = Counter(e["relation"] for e in all_edges)

    stats = {
        "extractor": extractor,
        "documents": len(files),
        "nodes": len(all_nodes),
        "edges": len(all_edges),
        "node_labels": dict(node_labels.most_common()),
        "entity_types": dict(entity_types.most_common()),
        "relations": dict(relations.most_common()),
        "graph_build_time_seconds": build_time,
        "graph_file_bytes": GRAPH_FILE.stat().st_size,
        "per_document": per_document,
    }

    GRAPH_STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Documents : {len(files)}")
    print(f"Nodes     : {len(all_nodes)}  {dict(node_labels)}")
    print(f"Edges     : {len(all_edges)}")
    print(f"Entity types: {dict(entity_types.most_common(10))}")
    print(f"Top relations: {dict(relations.most_common(15))}")
    print(f"Build time: {build_time:.2f}s")
    print(f"\nSaved {GRAPH_FILE}, {GRAPH_STATS_FILE}, graph.graphml and {NEO4J_EXPORT_DIR}/")


if __name__ == "__main__":
    main()
