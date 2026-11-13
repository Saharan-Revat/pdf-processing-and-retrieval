import json
from pathlib import Path


SEARCH_TERMS = [
    "one hundred twenty",
    "PRODUCT WARRANTY",
    "substantial accordance with the Documentation"
]

SEARCH_DIRS = [
    Path("data/parsed"),
    Path("data/cleaned"),
    Path("data/chunks")
]


def search_json(obj, term):
    """Recursively search all strings inside JSON."""

    matches = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            matches.extend(search_json(value, term))

    elif isinstance(obj, list):
        for item in obj:
            matches.extend(search_json(item, term))

    elif isinstance(obj, str):
        if term.lower() in obj.lower():
            matches.append(obj)

    return matches


def search_directory(directory, term):

    results = []

    for file_path in directory.rglob("*"):

        if not file_path.is_file():
            continue

        # Only inspect Changepoint
        if "CHANGEPOINT" not in file_path.name.upper():
            continue

        try:

            if file_path.suffix.lower() == ".json":

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    data = json.load(f)

                matches = search_json(data, term)

            else:

                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                matches = (
                    [text]
                    if term.lower() in text.lower()
                    else []
                )

            if matches:
                results.append(
                    (file_path, matches)
                )

        except Exception as e:
            print(
                f"Could not read {file_path}: {e}"
            )

    return results


if __name__ == "__main__":

    for term in SEARCH_TERMS:

        print("\n" + "=" * 80)
        print(f"SEARCHING FOR: {term}")
        print("=" * 80)

        for directory in SEARCH_DIRS:

            print(f"\n{directory}/")

            results = search_directory(
                directory,
                term
            )

            if not results:
                print("  NOT FOUND")
                continue

            print("  FOUND")

            for file_path, matches in results:

                print(f"\n  File: {file_path}")

                for match in matches[:2]:

                    normalized = " ".join(
                        match.split()
                    )

                    position = normalized.lower().find(
                        term.lower()
                    )

                    start = max(
                        0,
                        position - 200
                    )

                    end = min(
                        len(normalized),
                        position + 500
                    )

                    print(
                        " ",
                        normalized[start:end]
                    )