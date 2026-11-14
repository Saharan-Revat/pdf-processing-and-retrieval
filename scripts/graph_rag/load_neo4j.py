"""
Load data/knowledge_graph/graph.json into Neo4j (no APOC required).

    pip install neo4j
    # Neo4j Desktop / Docker:
    #   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
    export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=password
    python scripts/graph_rag/load_neo4j.py            # add --reset to wipe first

Then open http://localhost:7474 and try, e.g.

    // one contract's entity graph
    MATCH (e:Entity)-[r]->(x:Entity)
    WHERE e.document_id STARTS WITH 'DYNAMEX' AND type(r) <> 'CO_OCCURS_WITH'
    RETURN e, r, x LIMIT 150

    // multi-hop evidence path: question entity -> definition -> chunk
    MATCH p = (t:Entity {name:'Royalty Rate'})-[:DEFINED_IN|MENTIONED_IN]->(c:Chunk)
              -[:REFERS_TO]->(s:Section)<-[:CONTAINS_SECTION]-(c2:Chunk)
    RETURN p LIMIT 25

    // aliases resolved for the parties
    MATCH (p:Entity) WHERE size(p.aliases) > 0 RETURN p.name, p.aliases, p.document_id
"""

import argparse
import json
import os
import re
from collections import defaultdict

from kg_common import GRAPH_FILE

BATCH = 1000


def safe_label(text):
    text = re.sub(r"[^A-Za-z0-9_]", "_", text or "Other")
    return text if text and not text[0].isdigit() else f"T_{text}"


def batches(items, size=BATCH):
    for i in range(0, len(items), size):
        yield items[i: i + size]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete all existing nodes first")
    parser.add_argument("--skip-cooccurrence", action="store_true",
                        help="Do not load the weak CO_OCCURS_WITH edges (cleaner visualisation)")
    args = parser.parse_args()

    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    graph = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:

        if args.reset:
            print("Deleting existing graph ...")
            session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")

        session.run("CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")

        # ---------- nodes, grouped by label set ---------------------
        groups = defaultdict(list)
        for n in graph["nodes"]:
            labels = ["KGNode", safe_label(n["label"])]
            if n["label"] == "Entity":
                labels.append(safe_label(n.get("entity_type")))
            props = {k: v for k, v in n.items() if k != "match_keys" and v not in (None, "")}
            groups[":".join(labels)].append(props)

        for labels, rows in groups.items():
            for chunk in batches(rows):
                session.run(
                    f"UNWIND $rows AS row MERGE (n:KGNode {{id: row.id}}) SET n += row, n:{labels}",
                    rows=chunk,
                )
            print(f"  nodes :{labels:<35} {len(rows)}")

        # ---------- relationships, grouped by type ------------------
        by_type = defaultdict(list)
        for e in graph["edges"]:
            if args.skip_cooccurrence and e["relation"] == "CO_OCCURS_WITH":
                continue
            by_type[safe_label(e["relation"])].append({
                "source": e["source"], "target": e["target"],
                "props": {k: v for k, v in e.items() if k not in ("source", "target", "relation") and v not in (None, "")},
            })

        for rel_type, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            for chunk in batches(rows):
                session.run(
                    f"UNWIND $rows AS row "
                    f"MATCH (a:KGNode {{id: row.source}}), (b:KGNode {{id: row.target}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) SET r += row.props",
                    rows=chunk,
                )
            print(f"  rels  :{rel_type:<35} {len(rows)}")

    driver.close()
    print(f"\nLoaded {len(graph['nodes'])} nodes into {uri}")


if __name__ == "__main__":
    main()
