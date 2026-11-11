from pathlib import Path
from docling.document_converter import DocumentConverter

INPUT_DIR = Path("dataset/selected_pdfs")
OUTPUT_DIR = Path("data/parsed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()

pdfs = sorted(INPUT_DIR.glob("*.[pP][dD][fF]"))

print(f"Found {len(pdfs)} PDFs")

for i, pdf_path in enumerate(pdfs, 1):
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"

    if output_path.exists():
        print(f"[{i}/{len(pdfs)}] Already parsed: {pdf_path.name}")
        continue

    print(f"[{i}/{len(pdfs)}] Parsing: {pdf_path.name}")

    try:
        result = converter.convert(pdf_path)
        result.document.save_as_json(output_path)
    except Exception as e:
        print(f"ERROR: {pdf_path.name}")
        print(e)

print("Parsing complete.")
