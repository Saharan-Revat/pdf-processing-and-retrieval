# Vector Retrieval and Vector RAG

This directory contains the dense vector retrieval and Vector RAG implementation for the multi-hop legal document retrieval project.

The pipeline uses the cleaned and chunked CUAD contract corpus produced by the document-processing pipeline. The same evaluation questions and gold-evidence mapping used for the BM25 baseline are reused so that BM25 and Vector retrieval can be compared under the same experimental conditions.

## Pipeline

The Vector RAG pipeline consists of:

```text
Cleaned Contract Chunks
        |
        v
SentenceTransformer Embeddings
        |
        v
FAISS Vector Index
        |
        v
Question Embedding
        |
        v
Cosine Similarity Retrieval
        |
        v
Top-K Contract Chunks
        |
        v
Question-Focused Evidence Extraction
        |
        v
FLAN-T5 Base
        |
        v
Generated Answer + Source Attribution
```

## Dataset

The experiment uses:

- 17 CUAD contract documents
- 651 processed document chunks
- 100 evaluation questions from NLLP-CUAD-MultiHop
- 25 questions for each reasoning depth:
  - 2-hop: 25
  - 3-hop: 25
  - 4-hop: 25
  - 5-hop: 25

The evaluation uses the same corrected chunks and evaluation questions as the BM25 baseline.

## Models

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding configuration:

- Embedding dimension: 384
- Embeddings are L2-normalized
- Similarity metric: cosine similarity
- FAISS `IndexFlatIP` is used over normalized vectors

With normalized embeddings, inner-product ranking is equivalent to cosine-similarity ranking.

### Generation Model

```text
google/flan-t5-base
```

FLAN-T5 Base is used as the generation component of the Vector RAG baseline.

Generation configuration:

- Deterministic generation
- `do_sample=False`
- Beam search: 4 beams
- Maximum generated tokens: 50
- Maximum model input length: 512 tokens

## Vector Index

The vector index is constructed from all 651 document chunks.

Run:

```bash
python scripts/vector_rag/build_vector_index.py
```

This creates:

```text
data/vector_index/
├── chunks.faiss
├── metadata.json
└── index_info.json
```

`metadata.json` preserves information such as chunk ID, document ID, source file, pages, sections, and chunk text.

## Semantic Retrieval

Vector retrieval is implemented in:

```text
vector_retrieval.py
```

For each question:

1. The question is converted into a normalized 384-dimensional embedding.
2. The source contract is identified.
3. All chunks belonging to that contract are considered.
4. Cosine similarity is calculated between the question and contract chunks.
5. Chunks are ranked by similarity.
6. The Top-K chunks are returned.

This within-document setup is also used by the BM25 baseline, providing a controlled comparison between lexical and semantic retrieval.

Test semantic retrieval with:

```bash
python scripts/vector_rag/vector_retrieval.py
```

## Retrieval Evaluation

Vector retrieval is evaluated using the same 100 questions and gold-passage mapping procedure as BM25.

Run:

```bash
python scripts/vector_rag/evaluate_vector.py
```

The following values of K are evaluated:

```text
K = 1, 3, 5, 10
```

Metrics include:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Full Evidence Recovery
- Retrieval latency

Results are saved to:

```text
results/vector/vector_results.csv
```

## Retrieval Results

Overall Vector retrieval performance:

| K | Precision@K | Recall@K | MRR | Full Evidence Recovery |
|---:|---:|---:|---:|---:|
| 1 | 0.510 | 0.153 | 0.621 | 0.020 |
| 3 | 0.320 | 0.281 | 0.621 | 0.040 |
| 5 | 0.260 | 0.365 | 0.621 | 0.040 |
| 10 | 0.191 | 0.519 | 0.621 | 0.140 |

Average Vector retrieval latency was approximately:

```text
14.29 ms/query
```

Run the summary script with:

```bash
python scripts/vector_rag/summarize_vector.py
```

## Retrieval Results by Reasoning Depth

At K=10:

| Reasoning Depth | Precision@10 | Recall@10 | MRR | Full Evidence Recovery@10 |
|---|---:|---:|---:|---:|
| 2-hop | 0.128 | 0.493 | 0.519 | 0.120 |
| 3-hop | 0.216 | 0.620 | 0.651 | 0.280 |
| 4-hop | 0.204 | 0.533 | 0.690 | 0.160 |
| 5-hop | 0.216 | 0.429 | 0.623 | 0.000 |
| Overall | 0.191 | 0.519 | 0.621 | 0.140 |

Generate the report table with:

```bash
python scripts/vector_rag/create_vector_table.py
```

## BM25 Comparison

The same evaluation framework is used for both BM25 and Vector retrieval.

Overall K=10 comparison:

| Retriever | Precision@10 | Recall@10 | MRR | Full Evidence Recovery@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.222 | 0.617 | 0.788 | 0.200 |
| Vector | 0.191 | 0.519 | 0.621 | 0.140 |

In this experiment, BM25 produced stronger overall retrieval performance than the MiniLM dense-vector baseline.

This may be related to characteristics of legal contracts such as exact terminology, defined terms, entity names, numerical values, dates, and section references, which can favor lexical retrieval.

The comparison can be reproduced with:

```bash
python scripts/compare_retrievers.py
```

## Vector RAG

The complete Vector RAG implementation is contained in:

```text
vector_rag.py
```

The generation pipeline uses the Top-5 semantic retrieval results.

Because FLAN-T5 Base has a limited context window, complete 512-token chunks are not directly concatenated into the generation prompt.

Instead, a question-focused evidence extraction step selects a short region from each retrieved chunk based on overlap with the question.

The resulting evidence is provided to FLAN-T5 together with instructions to answer only from retrieved contract evidence.

The generated response is accompanied by the retrieved chunk IDs and page metadata for source attribution.

Test a single Vector RAG question with:

```bash
python scripts/vector_rag/vector_rag.py
```

## Vector RAG Evaluation

The final Vector RAG experiment evaluates all 100 multi-hop questions.

Run:

```bash
python scripts/vector_rag/evaluate_vector_rag.py
```

The experiment uses:

```text
Top-K retrieval = 5
Questions = 100
Questions per hop = 25
```

The RAG evaluation records:

- Generated answer
- Gold answer
- Gold Answer Match
- Whether any mapped gold evidence was retrieved
- Whether the complete mapped evidence set was retrieved
- Number of relevant chunks retrieved
- Retrieved chunk IDs
- Retrieval latency
- Generation latency

The primary answer metric is a normalized **Gold Answer Match Rate**. A generated response is counted as matching when the complete normalized gold answer occurs in the generated response.

## Vector RAG Results

| Reasoning Depth | Questions | Correct Answers | Gold Answer Match | Any Gold Evidence | Full Gold Evidence |
|---|---:|---:|---:|---:|---:|
| 2-hop | 25 | 1 | 4.0% | 68.0% | 8.0% |
| 3-hop | 25 | 2 | 8.0% | 88.0% | 4.0% |
| 4-hop | 25 | 0 | 0.0% | 84.0% | 4.0% |
| 5-hop | 25 | 0 | 0.0% | 80.0% | 0.0% |
| Overall | 100 | 3 | 3.0% | 80.0% | 4.0% |

Conditional generation results:

```text
Gold Answer Match when any gold evidence was retrieved:
3 / 80 = 3.8%

Gold Answer Match when the full gold evidence set was retrieved:
1 / 4 = 25.0%
```

The full-evidence subset contains only four questions, so the 25% value should be interpreted descriptively rather than as a statistically robust estimate.

## Interpretation

The dense retriever frequently recovered some relevant evidence, with at least one mapped gold chunk appearing in the Top-5 results for 80% of the evaluation questions.

However, complete evidence recovery was substantially more difficult. Only 4% of questions had their complete mapped evidence set available within the Top-5 retrieved chunks.

This limitation becomes particularly visible for deeper multi-hop questions. No 5-hop question achieved complete evidence recovery in the final Vector RAG experiment.

The generation experiment also shows that retrieval is not the only limitation. FLAN-T5 Base frequently failed to produce the gold answer even when partial relevant evidence was available.

The results therefore identify two separate failure modes:

1. **Retrieval failure:** the complete evidence chain required for the multi-hop question is not recovered.
2. **Generation/reasoning failure:** relevant evidence is retrieved, but the lightweight generation model does not correctly derive or extract the expected answer.

These results provide a baseline for comparison with the Graph RAG approach.

## Result Files

Vector retrieval outputs:

```text
results/vector/
├── vector_results.csv
├── vector_summary.csv
└── vector_report_table.csv
```

Vector RAG outputs:

```text
results/vector/
├── vector_rag_results.csv
├── vector_rag_summary.csv
└── vector_rag_report_table.csv
```

Plots:

```text
results/vector/plots/
├── vector_precision_by_hop.png
├── vector_recall_by_hop.png
├── vector_evidence_recovery.png
├── vector_overall_performance.png
└── vector_rag_by_hop.png
```

## Scripts

```text
scripts/vector_rag/
├── build_vector_index.py
├── vector_retrieval.py
├── evaluate_vector.py
├── summarize_vector.py
├── create_vector_table.py
├── plot_vector_results.py
├── vector_rag.py
├── evaluate_vector_rag.py
├── summarize_vector_rag.py
├── create_vector_rag_table.py
└── plot_vector_rag_results.py
```

## Reproducing the Experiments

From the repository root, run:

```bash
# 1. Build vector embeddings and FAISS index
python scripts/vector_rag/build_vector_index.py

# 2. Test semantic retrieval
python scripts/vector_rag/vector_retrieval.py

# 3. Evaluate semantic retrieval on 100 questions
python scripts/vector_rag/evaluate_vector.py

# 4. Summarize retrieval performance
python scripts/vector_rag/summarize_vector.py

# 5. Generate retrieval report table
python scripts/vector_rag/create_vector_table.py

# 6. Generate retrieval plots
python scripts/vector_rag/plot_vector_results.py

# 7. Test Vector RAG generation
python scripts/vector_rag/vector_rag.py

# 8. Evaluate Vector RAG on all 100 questions
python scripts/vector_rag/evaluate_vector_rag.py

# 9. Summarize Vector RAG
python scripts/vector_rag/summarize_vector_rag.py

# 10. Generate Vector RAG report table
python scripts/vector_rag/create_vector_rag_table.py

# 11. Generate Vector RAG plot
python scripts/vector_rag/plot_vector_rag_results.py

# 12. Compare BM25 and Vector retrieval
python scripts/compare_retrievers.py
```

## Dependencies

The primary Python dependencies are:

```text
sentence-transformers
faiss-cpu
transformers
sentencepiece
torch
numpy
pandas
matplotlib
```

The Hugging Face models are downloaded automatically when first used and should not be committed to the repository.

## Notes

- All scripts should be run from the repository root.
- The Vector and BM25 experiments use the same corrected document chunks and evaluation questions.
- Vector retrieval results should be interpreted separately from downstream generation quality.
- The low downstream answer-match rate reflects both incomplete multi-hop evidence recovery and limitations of the lightweight FLAN-T5 Base generator.
- The Graph RAG branch can use the same evaluation questions and gold-evidence framework for controlled comparison.