# Knowledge Graph and Graph RAG (Ziyad)

This is the third retrieval representation in the project, after BM25 and Vector RAG:

```
chunks (data/chunks) → entity + SPO-triple extraction → normalised knowledge graph
      → (Neo4j)  +  entity matching → bounded graph traversal → evidence chunks → LLM answer
```

Every entity, relation and statement keeps a link to the chunk (and pages) it came from, so each retrieved passage can be traced back through an explicit **evidence path**. For example:

```
'Section 3.1' -CONTAINS_SECTION- chunk 1
'Royalty Rate' -DEFINED_IN- chunk 12
```

## How to run

Run these from the repository root. Steps 1–4 take a few seconds on a laptop and need no GPU, no API key and no Neo4j.

```bash
pip install networkx pandas matplotlib rank-bm25          # neo4j only for step 5

python scripts/graph_rag/kg_extraction.py                 # 1. chunks  -> triples (rules)
python scripts/graph_rag/build_graph.py                   # 2. triples -> graph.json (+ Neo4j CSV, GraphML)
python scripts/graph_rag/evaluate_graph.py                # 3. retrieval metrics (same protocol as BM25 / Vector)
python scripts/compare_retrievers.py                      #    BM25 vs Vector vs Graph table
python scripts/graph_rag/plot_graph_results.py            # 4. plots + knowledge-graph figure

python scripts/graph_rag/load_neo4j.py --reset            # 5. optional: load into Neo4j for exploration
python scripts/graph_rag/evaluate_graph_rag.py            # 6. QA with the shared flan-t5 generator
```

Step 6 imports the generator from `scripts/vector_rag/vector_rag.py`, so it needs the same environment as Vector RAG (`transformers`, `torch`, `sentence-transformers`, `faiss`). It uses the same prompt, the same top-5 evidence budget, the same 100-question sample and the same answer-matching rule as `evaluate_vector_rag.py`, so retrieval is the only variable that changes.

### Using an LLM for extraction (as in the proposal)

```bash
# Local and free, with Ollama
ollama pull qwen2.5:7b-instruct
python scripts/graph_rag/kg_extraction.py --extractor both

# Any OpenAI-compatible API (OpenAI, Groq, Together, vLLM, ...)
export LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=sk-... LLM_MODEL=gpt-4o-mini
python scripts/graph_rag/kg_extraction.py --extractor both
```

Then re-run steps 2–4.

- `--extractor both` keeps the rule-based structural skeleton (sections, defined terms, aliases) and adds the LLM's semantic triples.
- LLM triples whose subject or object does not occur in the source chunk are dropped. This filters hallucinations and guarantees every triple has provenance.
- Responses are cached in `data/knowledge_graph/llm_cache/`, so an interrupted run resumes where it stopped. The 651 chunks mean 651 LLM calls.

## Files

| Script | What it does |
|---|---|
| `kg_common.py` | Paths; entity and section normalisation; re-uses the BM25 loaders and gold-mapping functions |
| `kg_extraction.py` | `rules`, `llm` or `both` extractor that produces `data/knowledge_graph/triples/*.json` |
| `build_graph.py` | Normalisation, alias merging (union-find), graph assembly, Neo4j CSV and GraphML export |
| `load_neo4j.py` | Loads `graph.json` into Neo4j (no APOC needed); example Cypher queries are in its docstring |
| `graph_retrieval.py` | Entity and section matching, Personalised PageRank traversal, chunk ranking, evidence paths |
| `evaluate_graph.py` | Precision, Recall, MRR and Full Evidence Recovery @ {1, 3, 5, 10}, by hop |
| `evaluate_graph_rag.py` | End-to-end QA with the shared generator |
| `plot_graph_results.py` | Four-way comparison plots and a knowledge-graph excerpt figure |

## Graph schema

```
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT]->(:Chunk)
(:Entity)-[:MENTIONED_IN]->(:Chunk)
(:Entity)-[:DEFINED_IN]->(:Chunk)
(:Entity)-[:ALSO_KNOWN_AS | HAS_DURATION | HAS_AMOUNT | HAS_PERCENTAGE | HAS_DATE |
           INCORPORATED_IN | OBLIGATION_INVOLVES | CO_OCCURS_WITH | <LLM predicates>]->(:Entity)
(:Entity)-[:DEFINED_AS | MUST | MUST_NOT | MAY | WARRANTS]->(:Statement)-[:STATED_IN]->(:Chunk)
(:Chunk)-[:CONTAINS_SECTION]->(:Section)        (:Chunk)-[:REFERS_TO]->(:Section)
(:Section)-[:PART_OF]->(:Section)               e.g. 3.1 -> 3
```

Entity types are Party, Organization, DefinedTerm, Duration, Amount, Percentage, Date, Location, plus the LLM types. Each node has a `chunk_ids` / `document_id` for provenance.

**Normalisation:**
- Names are lower-cased, and quotes, articles, possessives and corporate suffixes are removed. This makes "Dynamex Inc.", "DYNAMEX" and "Dynamex's" one node.
- `ALSO_KNOWN_AS` aliases are merged with union-find. This makes "PUROLATOR COURIER LTD." and "Purolator" one node.

**Current graph (rule extractor, 17 contracts, 651 chunks):**
- 3,286 nodes: 945 entities, 947 statements, 726 sections, 651 chunks and 17 documents.
- 11,944 edges.
- Extraction takes about 2 s and the graph build 0.2 s.

See `data/knowledge_graph/graph_stats.json` for per-document counts.

## Retrieval method

Retrieval is per contract, the same as the baselines.

1. **Entity matching.** Entity names and aliases are found in the question by exact phrase match, or by partial match when all content tokens of a multi-word name appear. Explicit section numbers ("sections 3.1 and 3.2") are matched too. Each seed is weighted by IDF over chunks, so specific terms ("Royalty Rate") count for more than generic ones. When a shorter name sits inside a longer matched one ("Product" inside "Licensed Product"), the shorter one is down-weighted.
2. **Bounded traversal.** A Personalised PageRank (random walk with restart) runs from the seeds. With `ALPHA = 0.75` the restart probability is 0.25, so walks stay within about 3 hops of the question's entities. Edges are weighted: definitions ×3, section containment ×2, section references ×1.5, mentions ×1, chunk adjacency and co-occurrence ×0.3.
3. **Ranking.** Chunks are ranked by walk score divided by √degree, which stops chunks with many entities from dominating every question.
4. **Provenance.** For each returned chunk, the shortest path from a seed is reported.

There are two variants:
- **Graph** is pure Graph RAG: the seeds come only from the graph. If the question matches no entity (34% of questions), it falls back to word overlap with entity names.
- **Graph+BM25** is graph-enhanced retrieval: BM25's top 5 chunks are added as extra seeds, getting half of the restart weight. The graph then expands to the definitions, cross-referenced sections and neighbouring clauses that lexical matching misses.

## Results (rule extractor, k = 10, 100 questions with 25 per hop)

| Retriever | Precision@10 | Recall@10 | MRR | Full Evidence Recovery@10 | Latency |
|---|---|---|---|---|---|
| BM25 | 0.222 | 0.616 | **0.788** | 0.20 | 0.2 ms |
| Vector | 0.191 | 0.519 | 0.621 | 0.14 | 14 ms |
| Graph | 0.199 | 0.547 | 0.534 | 0.17 | 2.9 ms |
| **Graph+BM25** | **0.231** | **0.638** | 0.711 | **0.26** | 2.1 ms |

Full Evidence Recovery@10 by reasoning depth (all gold passages retrieved):

| Hops | BM25 | Vector | Graph | Graph+BM25 |
|---|---|---|---|---|
| 2 | 0.28 | 0.12 | 0.16 | **0.36** |
| 3 | 0.24 | 0.28 | **0.32** | 0.24 |
| 4 | **0.24** | 0.16 | 0.12 | **0.24** |
| 5 | 0.04 | 0.00 | 0.08 | **0.20** |

Main observations, for RQ3 and RQ4:

- **Graph-enhanced retrieval gives the best recall and evidence recovery.** It is highest of the four on Recall@10 and Full Evidence Recovery@10.
- **The gain is largest on 5-hop questions:** 0.20 compared with 0.04 for BM25 and 0.00 for Vector. Following definition and section links recovers the far-away passages that a single lexical or semantic match does not reach.
- **Pure Graph RAG beats Vector** on precision, recall and evidence recovery, **but not BM25**. With rule-based extraction, a third of the questions match no graph entity. That is the extraction-quality bottleneck the proposal anticipated, and LLM extraction should reduce it.
- **BM25 still gives the best first hit (MRR).** The graph spreads score over related chunks, which pushes the single best lexical match down. This is a precision-versus-coverage trade-off.

**Sensitivity check.** The defaults (`ALPHA = 0.75`, degree normalisation 0.5, BM25 weight 0.5) were set before evaluating. Across ALPHA ∈ {0.6, 0.75, 0.85} × normalisation ∈ {0, 0.5} × BM25 weight ∈ {0.3, 0.5, 0.7}, Graph+BM25 stayed in Recall@10 0.638–0.657 and Full Evidence Recovery 0.23–0.29. Both stay above BM25 in every setting, so the result does not depend on the exact parameters.

Per-question matched entities, retrieved chunks (`*` = gold) and evidence paths are in `results/graph/graph_retrieval_details.csv`.

## Limitations and next steps

- The committed graph comes from the **rule-based** extractor. Re-run with `--extractor both` for the LLM-based knowledge graph described in the proposal, and report both: that comparison is itself an ablation.
- The QA numbers (`evaluate_graph_rag.py`) still need to be produced in the Vector RAG environment. The flan-t5-base generator is weak (3/100 correct for Vector RAG), so answer accuracy will mostly reflect the generator. Retrieval metrics are the cleaner comparison.
- Manual inspection for the proposal's "structured knowledge evaluation" (precision of entities and triples on a sample, plus checking provenance) is not done yet. `data/knowledge_graph/triples/*.json` holds each triple with its evidence sentence, which makes that quick to do.
