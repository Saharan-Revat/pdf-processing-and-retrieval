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


def clean_document(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_id = input_path.stem
    source_file = f"{document_id}.pdf"

    elements = []
    current_section = None
    current_section_level = None

    # Traverse Docling body in actual reading order
    for child in data.get("body", {}).get("children", []):
        ref = child.get("$ref", "")

        # ---------- TEXT ----------
        if ref.startswith("#/texts/"):
            index = int(ref.split("/")[-1])
            item = data["texts"][index]

            text = item.get("text", "").strip()
            label = item.get("label")

            if not text:
                continue

            # Remove repeated document furniture
            if label in {"page_header", "page_footer"}:
                continue

            if label == "section_header":
                current_section = text
                current_section_level = item.get("level")

            elements.append({
                "element_id": f"{document_id}::element::{len(elements)}",
                "document_id": document_id,
                "page_no": get_page_no(item),
                "section": current_section,
                "section_level": current_section_level,
                "label": label,
                "text": text,
                "source_file": source_file
            })

        # ---------- TABLE ----------
        elif ref.startswith("#/tables/"):
            index = int(ref.split("/")[-1])
            table = data["tables"][index]

            elements.append({
                "element_id": f"{document_id}::element::{len(elements)}",
                "document_id": document_id,
                "page_no": get_page_no(table),
                "section": current_section,
                "section_level": current_section_level,
                "label": "table",
                "table_text": json.dumps(
                    table.get("data", {}),
                    ensure_ascii=False
                ),
                "source_file": source_file
            })

    output = {
        "document_id": document_id,
        "source_file": source_file,
        "elements": elements
    }

    output_path = OUTPUT_DIR / f"{document_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return len(elements)


def main():
    files = sorted(INPUT_DIR.glob("*.json"))

    print(f"Found {len(files)} parsed documents")

    for i, path in enumerate(files, 1):
        count = clean_document(path)
        print(
            f"[{i}/{len(files)}] "
            f"{path.name} -> {count} elements"
        )

    print("Cleaning complete.")


if __name__ == "__main__":
    main()