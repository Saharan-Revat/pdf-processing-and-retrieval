import json
from pathlib import Path
import re
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher


CHUNKS_DIR = Path("data/chunks")
EVALUATION_DIR = Path("data/evaluation")


def load_chunks(chunks_dir):
    all_chunks = []

    json_files = list(chunks_dir.glob("*.json"))

    print(f"Found {len(json_files)} chunk files.")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])

        for chunk in chunks:
            all_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source_file": chunk["source_file"],
                    "sections": chunk.get("sections", []),
                    "page_numbers": chunk.get("page_numbers", []),
                    "text": chunk["text"],
                }
            )

    return all_chunks
def load_evaluation(evaluation_dir):
    questions = []

    jsonl_files = list(evaluation_dir.glob("*.jsonl"))

    print(f"Found {len(jsonl_files)} evaluation files.")

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)
                questions.append(item)

    return questions

def tokenize(text):
    """
    Simple tokenizer for BM25.

    Converts text to lowercase and extracts
    alphanumeric word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())
def build_bm25_indexes(chunks):
    """
    Build a separate BM25 index for each document.
    """

    documents = {}

    # Group chunks by document
    for chunk in chunks:
        document_id = chunk["document_id"]

        if document_id not in documents:
            documents[document_id] = []

        documents[document_id].append(chunk)

    indexes = {}

    for document_id, document_chunks in documents.items():

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in document_chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        indexes[document_id] = {
            "bm25": bm25,
            "chunks": document_chunks
        }

    return indexes


def get_question_document(question):
    """
    Determine the source document from the first gold passage.
    """

    gold_passages = question["gold_passages"]

    if not gold_passages:
        return None

    gold_chunk_id = gold_passages[0]["chunk_id"]

    document_id = gold_chunk_id.rsplit("::", 1)[0]

    return document_id


def retrieve_bm25(question, indexes, top_k=5):
    """
    Retrieve the top-k BM25 chunks for a question.
    """

    document_id = get_question_document(question)

    if document_id not in indexes:
        raise ValueError(
            f"Document not found in BM25 indexes: {document_id}"
        )

    index_data = indexes[document_id]

    bm25 = index_data["bm25"]
    document_chunks = index_data["chunks"]

    tokenized_query = tokenize(question["question"])

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for rank, idx in enumerate(ranked_indices[:top_k], start=1):

        chunk = document_chunks[idx]

        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "pages": chunk["page_numbers"],
            "text": chunk["text"]
        })

    return results

def text_similarity(text1, text2):
    """
    Calculate normalized text similarity.
    """
    text1 = " ".join(text1.lower().split())
    text2 = " ".join(text2.lower().split())

    return SequenceMatcher(None, text1, text2).ratio()

def map_gold_passage_to_chunks(gold_passage, document_chunks):
    """
    Find the project chunk(s) that best correspond to a gold passage.

    Uses exact substring matching first, followed by
    text similarity when chunk boundaries differ.
    """

    gold_text = " ".join(gold_passage["text"].lower().split())

    matches = []

    for chunk in document_chunks:

        chunk_text = " ".join(chunk["text"].lower().split())

        # Exact containment
        if gold_text in chunk_text or chunk_text in gold_text:
            similarity = 1.0
        else:
            similarity = text_similarity(gold_text, chunk_text)

        matches.append({
            "chunk_id": chunk["chunk_id"],
            "similarity": similarity,
            "text": chunk["text"]
        })

    matches.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return matches

def search_chunks_for_text(document_chunks, search_text):
    search_text = search_text.lower()

    matches = []

    for chunk in document_chunks:
        if search_text in chunk["text"].lower():
            matches.append(chunk)

    return matches

def normalize_text(text):
    """
    Normalize text for basic answer matching.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_answer_coverage(question, indexes):
    """
    Check whether the gold answer appears anywhere
    in the processed chunks for the question's contract.
    """

    document_id = get_question_document(question)

    if document_id not in indexes:
        return False, []

    document_chunks = indexes[document_id]["chunks"]

    gold_answer = normalize_text(question["gold_answer"])

    matching_chunks = []

    for chunk in document_chunks:

        chunk_text = normalize_text(chunk["text"])

        if gold_answer in chunk_text:
            matching_chunks.append(chunk["chunk_id"])

    return len(matching_chunks) > 0, matching_chunks

def passage_coverage(gold_text, chunk_text):
    """
    Measure how much of the gold passage vocabulary
    appears in a project chunk.
    """

    gold_tokens = set(tokenize(gold_text))
    chunk_tokens = set(tokenize(chunk_text))

    if not gold_tokens:
        return 0.0

    overlap = gold_tokens.intersection(chunk_tokens)

    return len(overlap) / len(gold_tokens)



def find_best_gold_mapping(gold_passage, document_chunks):
    """
    Find the project chunk with the highest token coverage
    for a gold passage.
    """

    best_chunk = None
    best_score = 0.0

    for chunk in document_chunks:

        score = passage_coverage(
            gold_passage["text"],
            chunk["text"]
        )

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score


def evaluate_gold_passage_coverage(questions, indexes, threshold=0.5):

    total_passages = 0
    mapped_passages = 0

    question_results = []

    for question in questions:

        document_id = get_question_document(question)

        if document_id not in indexes:
            continue

        document_chunks = indexes[document_id]["chunks"]

        mappings = []

        for gold_passage in question["gold_passages"]:

            total_passages += 1

            best_chunk, score = find_best_gold_mapping(
                gold_passage,
                document_chunks
            )

            is_mapped = score >= threshold

            if is_mapped:
                mapped_passages += 1

            mappings.append({
                "gold_id": gold_passage["chunk_id"],
                "project_chunk_id":
                    best_chunk["chunk_id"] if best_chunk else None,
                "coverage": score,
                "mapped": is_mapped
            })

        question_results.append({
            "question_id": question["question_id"],
            "hop_count": question["hop_count"],
            "mappings": mappings
        })

    return {
        "total_passages": total_passages,
        "mapped_passages": mapped_passages,
        "question_results": question_results
    }

if __name__ == "__main__":

    chunks = load_chunks(CHUNKS_DIR)
    questions = load_evaluation(EVALUATION_DIR)

    indexes = build_bm25_indexes(chunks)

    results = evaluate_gold_passage_coverage(
        questions,
        indexes,
        threshold=0.5
    )

    print("\n" + "=" * 80)
    print("UNMAPPED GOLD PASSAGES")
    print("=" * 80)

    unmapped_count = 0

    for question_result in results["question_results"]:

        # Find original question
        original_question = next(
            q for q in questions
            if q["question_id"] == question_result["question_id"]
        )

        for mapping in question_result["mappings"]:

            if mapping["mapped"]:
                continue

            unmapped_count += 1

            # Find original gold passage
            gold_passage = next(
                p for p in original_question["gold_passages"]
                if p["chunk_id"] == mapping["gold_id"]
            )

            print("\n" + "=" * 80)
            print(f"UNMAPPED #{unmapped_count}")
            print("=" * 80)

            print("Question ID:",
                  original_question["question_id"])

            print("Hop:",
                  original_question["hop_count"])

            print("\nQuestion:")
            print(original_question["question"])

            print("\nGold answer:")
            print(original_question["gold_answer"])

            print("\nGold passage ID:")
            print(gold_passage["chunk_id"])

            print("\nGold passage:")
            print(gold_passage["text"])

            print("\nBest project chunk:")
            print(mapping["project_chunk_id"])

            print(
                f"\nCoverage score: "
                f"{mapping['coverage']:.4f}"
            )

            # Print best project chunk text
            document_id = get_question_document(
                original_question
            )

            document_chunks = indexes[
                document_id
            ]["chunks"]

            best_chunk = next(
                (
                    c for c in document_chunks
                    if c["chunk_id"]
                    == mapping["project_chunk_id"]
                ),
                None
            )

            if best_chunk:

                print("\nBest project chunk text:")
                print(best_chunk["text"])

            print("\n" + "-" * 80)

    print(
        f"\nTotal unmapped passages: "
        f"{unmapped_count}"
    )