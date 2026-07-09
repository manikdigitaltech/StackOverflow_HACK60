import json
from collections import Counter
from pathlib import Path


FILES = [
    Path("finetune_data/train.jsonl"),
    Path("finetune_data/validation.jsonl"),
    Path("finetune_data/test.jsonl"),
]

FORBIDDEN_REVIEW_SOURCES = {"iclr_2017", "acl_2017"}

OLD_TEMPLATE_PHRASES = [
    "The paper addresses a relevant and meaningful research problem.",
    "The proposed work attempts to contribute to the existing literature.",
    "The method appears to target an identifiable limitation in prior work.",
    "The manuscript presents a concrete approach that can be evaluated experimentally.",
    "The paper would benefit from stronger experimental evidence.",
    "The comparison with competitive baseline methods should be clearer and more systematic.",
    "Some claims require deeper justification and stronger empirical support.",
    "The discussion of limitations and failure cases could be improved.",
    "The contribution appears moderate. The idea is connected to an important research direction",
    "No detailed reviewer comments are available.",
]

REVIEW_DATASET_MARKERS = [
    "Write a scientific peer review for the given research paper.",
    "reviewer-style academic language",
    "### Review:",
    "ICLR committee final decision",
]

REQUIRED_KEYS = {"source", "original_split", "paper_id", "prompt", "completion"}
MAX_FAILURE_DETAILS_PER_FILE = 20


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def validate_file(path):
    print("\nChecking:", path)

    if not path.exists():
        print("FAILED: file is missing")
        return 1

    rows = load_rows(path)
    failures = 0
    printed_failures = 0

    def record_failure(message, detail=None):
        nonlocal failures, printed_failures
        failures += 1
        if printed_failures < MAX_FAILURE_DETAILS_PER_FILE:
            print(message)
            if detail:
                print(detail)
            printed_failures += 1

    print("Rows:", len(rows))
    print("Sources:", dict(Counter(row.get("source", "MISSING") for row in rows)))

    for row in rows:
        missing_keys = REQUIRED_KEYS - set(row)
        if missing_keys:
            record_failure(f"FAILED line {row['_line_number']}: missing keys {sorted(missing_keys)}")

        source = row.get("source")
        if source in FORBIDDEN_REVIEW_SOURCES:
            record_failure(f"FAILED line {row['_line_number']}: forbidden review dataset source {source!r}")

        text = "\n".join(str(row.get(key, "")) for key in ("prompt", "completion"))

        for phrase in OLD_TEMPLATE_PHRASES:
            if phrase in text:
                record_failure(
                    f"FAILED line {row['_line_number']}: old template phrase found",
                    f"Phrase: {phrase}",
                )

        for marker in REVIEW_DATASET_MARKERS:
            if marker in text:
                record_failure(
                    f"FAILED line {row['_line_number']}: review dataset marker found",
                    f"Marker: {marker}",
                )

    if failures == 0:
        print("PASSED: no old template rows or forbidden review dataset rows found")
    else:
        hidden = failures - printed_failures
        if hidden:
            print(f"... {hidden} additional failure details suppressed")
        print("FAILED checks:", failures)

    return failures


def main():
    total_failures = 0

    for path in FILES:
        total_failures += validate_file(path)

    if total_failures:
        raise SystemExit(1)

    print("\nAll fine-tuning data checks passed.")


if __name__ == "__main__":
    main()
