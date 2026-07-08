import json
from pathlib import Path


# ----------------------------------------------------
# Input clean files and output fine-tuning files
# ----------------------------------------------------
FILES = [
    (
        Path("peerread_processed/train/train_review_style_clean.jsonl"),
        Path("finetune_data/train.jsonl")
    ),
    (
        Path("peerread_processed/dev/dev_review_style_clean.jsonl"),
        Path("finetune_data/validation.jsonl")
    ),
    (
        Path("peerread_processed/test/test_review_style_clean.jsonl"),
        Path("finetune_data/test.jsonl")
    ),
]


def make_prompt(row):
    prompt = f"""### Instruction:
{row["instruction"]}

### Paper:
{row["input"]}

### Review:
"""
    return prompt


def main():
    Path("finetune_data").mkdir(exist_ok=True)

    for input_file, output_file in FILES:

        if not input_file.exists():
            print("Missing file:", input_file)
            continue

        count = 0

        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:

            for line in infile:
                if not line.strip():
                    continue

                row = json.loads(line)

                final_row = {
                    "prompt": make_prompt(row),
                    "completion": row["output"]
                }

                outfile.write(json.dumps(final_row, ensure_ascii=False) + "\n")
                count += 1

        print("Created:", output_file)
        print("Rows:", count)
        print("-" * 60)


if __name__ == "__main__":
    main()