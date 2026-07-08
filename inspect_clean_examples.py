import json
import random
from pathlib import Path


FILE = Path("peerread_processed/train/train_review_style_clean.jsonl")


def main():
    rows = []

    with open(FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    samples = random.sample(rows, min(3, len(rows)))

    for i, row in enumerate(samples, start=1):
        print("\n" + "=" * 80)
        print("EXAMPLE", i)
        print("=" * 80)
        print("\nINPUT PREVIEW:")
        print(row["input"][:700])
        print("\nOUTPUT:")
        print(row["output"])


if __name__ == "__main__":
    main()