import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("data/vector_index")
EVALUATION_DIR = Path("data/evaluation")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_vector_system():
    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_DIR / "chunks.faiss")
    )

    with open(
        INDEX_DIR / "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    print(f"Vectors: {index.ntotal}")
    print(f"Metadata records: {len(metadata)}")

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


def load_evaluation():
    questions = []

    files = list(EVALUATION_DIR.glob("*.jsonl"))

    for file_path in files:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                questions.append(
                    json.loads(line)
                )

    return questions


def get_question_document(question):
    """
    Extract source contract from the first gold passage.
    """

    gold_passages = question["gold_passages"]

    if not gold_passages:
        return None

    gold_id = gold_passages[0]["chunk_id"]

    return gold_id.rsplit("::", 1)[0]


def retrieve_vector(
    question,
    index,
    metadata,
    model,
    top_k=5,
    restrict_to_document=True
):
    """
    Retrieve top-k chunks using semantic similarity.

    When restrict_to_document=True, ranking is performed
    exactly over chunks from the question's source contract.
    """

    query_text = question["question"]

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if not restrict_to_document:

        scores, indices = index.search(
            query_embedding,
            top_k
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):
            if idx < 0:
                continue

            chunk = metadata[idx]

            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "pages": chunk.get("page_numbers", []),
                "text": chunk["text"]
            })

        return results

    # Exact within-document retrieval
    source_document = get_question_document(question)

    document_indices = [
        i
        for i, chunk in enumerate(metadata)
        if chunk["document_id"] == source_document
    ]

    if not document_indices:
        return []

    # Reconstruct stored normalized vectors
    document_vectors = np.vstack([
        index.reconstruct(i)
        for i in document_indices
    ]).astype("float32")

    # Cosine similarity because vectors are normalized
    similarities = (
        document_vectors @ query_embedding[0]
    )

    ranked_positions = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for rank, position in enumerate(
        ranked_positions,
        start=1
    ):
        metadata_index = document_indices[position]
        chunk = metadata[metadata_index]

        results.append({
            "rank": rank,
            "score": float(similarities[position]),
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "pages": chunk.get("page_numbers", []),
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":

    index, metadata, model = load_vector_system()

    questions = load_evaluation()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # Same first question used for BM25 testing
    question = questions[0]

    print("\n" + "=" * 80)
    print("TEST QUESTION")
    print("=" * 80)

    print("\nQuestion:")
    print(question["question"])

    print("\nGold answer:")
    print(question["gold_answer"])

    print("\nSource document:")
    print(get_question_document(question))

    results = retrieve_vector(
        question,
        index,
        metadata,
        model,
        top_k=5,
        restrict_to_document=True
    )

    print("\n" + "=" * 80)
    print("TOP 5 VECTOR RESULTS")
    print("=" * 80)

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Pages: "
            f"{result['pages']}"
        )

        print("Text:")

        print(
            result["text"][:700]
        )

        print("-" * 80)