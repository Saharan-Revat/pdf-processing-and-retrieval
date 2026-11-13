import json
from pathlib import Path


INPUT_DIR = Path("data/parsed")
OUTPUT_DIR = Path("data/cleaned")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_page_no(item):
    prov = item.get("prov", [])
    if prov:
        return prov[0].get("page_no")
    return None


def extract_table_text(table):
    """
    Convert Docling table data into retrieval-friendly plain text.
    Avoid storing bounding boxes and other layout metadata.
    """
    data = table.get("data", {})
    cells = data.get("table_cells", [])

    if not cells:
        return ""

    rows = {}

    for cell in cells:
        text = str(cell.get("text", "")).strip()

        if not text:
            continue

        row = cell.get("start_row_offset_idx", 0)
        col = cell.get("start_col_offset_idx", 0)

        rows.setdefault(row, [])
        rows[row].append((col, text))

    output_rows = []

    for row_index in sorted(rows):
        cells_in_row = sorted(
            rows[row_index],
            key=lambda x: x[0]
        )

        row_text = " | ".join(
            text for _, text in cells_in_row
        )

        if row_text:
            output_rows.append(row_text)

    return "\n".join(output_rows)


def clean_document(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_id = input_path.stem
    source_file = f"{document_id}.pdf"

    elements = []

    # ---------------------------------------------------------
    # TEXT
    # ---------------------------------------------------------
    #
    # Preserve all text objects produced by Docling instead of
    # relying only on direct body children. Some valid content
    # can otherwise disappear during cleaning.
    #
    current_section = None
    current_section_level = None

    for index, item in enumerate(data.get("texts", [])):

        text = item.get("text", "").strip()
        label = item.get("label")

        if not text:
            continue

        # Remove obvious document furniture only
        if label in {"page_header", "page_footer"}:
            continue

        if label == "section_header":
            current_section = text
            current_section_level = item.get("level")

        elements.append({
            "element_id":
                f"{document_id}::element::{len(elements)}",

            "document_id": document_id,
            "page_no": get_page_no(item),

            "section": current_section,
            "section_level": current_section_level,

            "label": label,
            "text": text,

            "source_file": source_file
        })

    # ---------------------------------------------------------
    # TABLES
    # ---------------------------------------------------------

    for table in data.get("tables", []):

        table_text = extract_table_text(table)

        if not table_text:
            continue

        elements.append({
            "element_id":
                f"{document_id}::element::{len(elements)}",

            "document_id": document_id,
            "page_no": get_page_no(table),

            "section": None,
            "section_level": None,

            "label": "table",
            "table_text": table_text,

            "source_file": source_file
        })

    # ---------------------------------------------------------
    # Sort approximately by page
    # ---------------------------------------------------------

    elements.sort(
        key=lambda x: (
            x["page_no"]
            if x["page_no"] is not None
            else 999999
        )
    )

    # Regenerate element IDs after sorting
    for i, element in enumerate(elements):
        element["element_id"] = (
            f"{document_id}::element::{i}"
        )

    output = {
        "document_id": document_id,
        "source_file": source_file,
        "elements": elements
    }

    output_path = OUTPUT_DIR / f"{document_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    return len(elements)


def main():

    files = sorted(INPUT_DIR.glob("*.json"))

    print(f"Found {len(files)} parsed documents")

    total_elements = 0

    for i, path in enumerate(files, 1):

        count = clean_document(path)

        total_elements += count

        print(
            f"[{i}/{len(files)}] "
            f"{path.name} -> {count} elements"
        )

    print()
    print("Cleaning complete.")
    print(f"Total elements: {total_elements}")


if __name__ == "__main__":
    main()