import argparse
import json
import os
from pathlib import Path

import ollama


TEST_FILE = Path("finetune_data/test.jsonl")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def load_examples(path, limit):
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python3 prepare_llama_finetune.py` first."
        )

    examples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            if len(examples) >= limit:
                break

    if not examples:
        raise SystemExit(f"No examples found in {path}.")

    return examples


def generate_review(model_name, prompt):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    except Exception as error:
        raise SystemExit(
            "Could not connect to Ollama. Start Ollama and make sure the model "
            f"`{model_name}` is available, then rerun this script.\n"
            f"Original error: {error}"
        ) from error

    return response["message"]["content"].strip()


def print_example(index, example, generated):
    print("\n" + "=" * 90)
    print(f"TEST EXAMPLE {index}")
    print("=" * 90)
    print("Source:", example.get("source", "unknown"))
    print("Paper ID:", example.get("paper_id", "unknown"))

    print("\n----- MODEL OUTPUT -----\n")
    print(generated)

    print("\n----- EXPECTED PEERREAD-BASED OUTPUT -----\n")
    print(example["completion"])


def main():
    parser = argparse.ArgumentParser(
        description="Test local Llama/Ollama model on PeerRead held-out examples."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name. Defaults to OLLAMA_MODEL or llama3.2.",
    )
    parser.add_argument(
        "--test-file",
        default=TEST_FILE,
        type=Path,
        help="JSONL test file created by prepare_llama_finetune.py.",
    )
    parser.add_argument(
        "--limit",
        default=3,
        type=int,
        help="Number of test examples to generate.",
    )
    args = parser.parse_args()

    examples = load_examples(args.test_file, args.limit)

    print("Testing model:", args.model)
    print("Test file:", args.test_file)
    print("Examples:", len(examples))

    for index, example in enumerate(examples, start=1):
        generated = generate_review(args.model, example["prompt"])
        print_example(index, example, generated)


if __name__ == "__main__":
    main()
