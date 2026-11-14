import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/vector_index")

INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    files = sorted(CHUNKS_DIR.glob("*.json"))

    print(f"Found {len(files)} chunk files.")

    for file_path in files:

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for chunk in data["chunks"]:

            chunks.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "source_file": chunk["source_file"],
                "page_numbers": chunk.get(
                    "page_numbers",
                    []
                ),
                "sections": chunk.get(
                    "sections",
                    []
                ),
                "text": chunk["text"]
            })

    return chunks


def main():

    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")

    print(f"\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nGenerating embeddings...")

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedding_time = (
        time.perf_counter() - start
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"\nEmbedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding generation time: "
        f"{embedding_time:.3f} seconds"
    )

    # Cosine similarity using normalized vectors
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"Vectors in FAISS index: "
        f"{index.ntotal}"
    )

    # Save index
    faiss.write_index(
        index,
        str(INDEX_DIR / "chunks.faiss")
    )

    # Save metadata
    with open(
        INDEX_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save experiment metadata
    experiment_info = {
        "model": MODEL_NAME,
        "num_chunks": len(chunks),
        "embedding_dimension":
            int(embeddings.shape[1]),
        "normalized_embeddings": True,
        "similarity": "cosine",
        "embedding_time_seconds":
            embedding_time
    }

    with open(
        INDEX_DIR / "index_info.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            experiment_info,
            f,
            indent=2
        )

    print("\nSaved:")
    print(
        INDEX_DIR / "chunks.faiss"
    )
    print(
        INDEX_DIR / "metadata.json"
    )
    print(
        INDEX_DIR / "index_info.json"
    )


if __name__ == "__main__":
    main()