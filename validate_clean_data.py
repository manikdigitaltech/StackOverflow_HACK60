import json
from pathlib import Path


FILES = [
    Path("peerread_processed/train/train_review_style_clean.jsonl"),
    Path("peerread_processed/dev/dev_review_style_clean.jsonl"),
    Path("peerread_processed/test/test_review_style_clean.jsonl"),
]

REQUIRED_KEYS = ["instruction", "input", "output"]

REQUIRED_SECTIONS = [
    "Summary:",
    "Strengths:",
    "Weaknesses:",
    "Novelty:",
    "Recommendation:",
]


def validate_file(file_path):
    print("\nChecking:", file_path)

    if not file_path.exists():
        print("File missing.")
        return

    total_rows = 0
    bad_rows = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print("Invalid JSON at line:", line_number)
                bad_rows += 1
                continue

            for key in REQUIRED_KEYS:
                if key not in row:
                    print(f"Missing key '{key}' at line {line_number}")
                    bad_rows += 1
                    break

            output = row.get("output", "")

            for section in REQUIRED_SECTIONS:
                if section not in output:
                    print(f"Missing section '{section}' at line {line_number}")
                    bad_rows += 1
                    break

    print("Total rows:", total_rows)
    print("Bad rows:", bad_rows)


def main():
    for file_path in FILES:
        validate_file(file_path)


if __name__ == "__main__":
    main()