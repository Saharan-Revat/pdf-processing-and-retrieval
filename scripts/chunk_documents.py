import json
from pathlib import Path
import tiktoken

INPUT_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("data/chunks")

MAX_TOKENS = 512
MIN_TOKENS = 100
OVERLAP_TOKENS = 64

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
encoding = tiktoken.get_encoding("cl100k_base")


def encode(text):
    return encoding.encode(text)


def token_count(text):
    return len(encode(text))


def element_text(element):
    if element["label"] == "table":
        return element.get("table_text", "")
    return element.get("text", "")


def split_text(text):
    tokens = encode(text)
    pieces = []
    start = 0

    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))

        # Decode, then re-encode to guarantee <= MAX_TOKENS
        piece = encoding.decode(tokens[start:end]).strip()
        piece_tokens = encode(piece)[:MAX_TOKENS]
        piece = encoding.decode(piece_tokens).strip()

        if piece:
            pieces.append(piece)

        if end >= len(tokens):
            break

        start = end - OVERLAP_TOKENS

    return pieces


def make_chunk(document_id, source_file, number, text, elements):
    pages = sorted({
        e["page_no"]
        for e in elements
        if e.get("page_no") is not None
    })

    sections = []
    for e in elements:
        section = e.get("section")
        if section and section not in sections:
            sections.append(section)

    return {
        "chunk_id": f"{document_id}::chunk::{number}",
        "document_id": document_id,
        "source_file": source_file,
        "sections": sections,
        "page_numbers": pages,
        "element_ids": list(dict.fromkeys(
            e["element_id"] for e in elements
        )),
        "token_count": token_count(text),
        "text": text
    }


def chunk_document(path):
    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)

    document_id = document["document_id"]
    source_file = document["source_file"]

    # Build section groups
    groups = []
    current_key = None
    current_elements = []

    for element in document["elements"]:
        key = (
            element.get("section"),
            element.get("section_level")
        )

        if current_elements and key != current_key:
            groups.append(current_elements)
            current_elements = []

        current_key = key
        current_elements.append(element)

    if current_elements:
        groups.append(current_elements)

    # Merge tiny section groups with the next group where possible
    merged_groups = []
    pending = []

    for group in groups:
        group_text = "\n\n".join(
            element_text(e).strip()
            for e in group
            if element_text(e).strip()
        )

        if not group_text:
            continue

        if pending:
            pending.extend(group)

            pending_text = "\n\n".join(
                element_text(e).strip()
                for e in pending
                if element_text(e).strip()
            )

            if token_count(pending_text) >= MIN_TOKENS:
                merged_groups.append(pending)
                pending = []

            continue

        if token_count(group_text) < MIN_TOKENS:
            pending = list(group)
        else:
            merged_groups.append(group)

    # If final tiny group remains, merge backward
    if pending:
        if merged_groups:
            merged_groups[-1].extend(pending)
        else:
            merged_groups.append(pending)

    # Create final chunks
    chunks = []

    for group in merged_groups:
        text = "\n\n".join(
            element_text(e).strip()
            for e in group
            if element_text(e).strip()
        )

        if token_count(text) <= MAX_TOKENS:
            chunks.append(
                make_chunk(
                    document_id,
                    source_file,
                    len(chunks),
                    text,
                    group
                )
            )
        else:
            for piece in split_text(text):
                chunks.append(
                    make_chunk(
                        document_id,
                        source_file,
                        len(chunks),
                        piece,
                        group
                    )
                )

    output = {
        "document_id": document_id,
        "source_file": source_file,
        "max_tokens": MAX_TOKENS,
        "min_tokens": MIN_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "chunks": chunks
    }

    with open(OUTPUT_DIR / path.name, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return chunks


def main():
    files = sorted(INPUT_DIR.glob("*.json"))
    print(f"Found {len(files)} cleaned documents")

    total_chunks = 0
    all_sizes = []

    for i, path in enumerate(files, 1):
        chunks = chunk_document(path)
        sizes = [c["token_count"] for c in chunks]

        total_chunks += len(chunks)
        all_sizes.extend(sizes)

        print(
            f"[{i}/{len(files)}] {path.name} -> "
            f"{len(chunks)} chunks | "
            f"avg={sum(sizes)/len(sizes):.1f} | "
            f"min={min(sizes)} | max={max(sizes)}"
        )

    print(f"\nTotal chunks: {total_chunks}")
    print(f"Overall average: {sum(all_sizes)/len(all_sizes):.1f}")
    print(f"Overall min: {min(all_sizes)}")
    print(f"Overall max: {max(all_sizes)}")


if __name__ == "__main__":
    main()