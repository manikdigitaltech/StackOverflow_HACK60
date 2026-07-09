import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TEST_FILE = Path("finetune_data/test.jsonl")
DEFAULT_BASE_MODEL = "distilgpt2"
DEFAULT_ADAPTER_DIR = Path("models/peer_review_lora")


def load_examples(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No examples found in {path}")
    return rows


def format_prompt(prompt):
    return f"{prompt.strip()}\n\nReview:\n"


def load_model(base_model, adapter_dir, local_files_only):
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def generate_review(
    tokenizer,
    model,
    prompt,
    max_input_tokens,
    max_new_tokens,
    temperature,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
):
    prompt_text = format_prompt(prompt)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        output_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def print_result(index, example, generated):
    print("\n" + "=" * 90)
    print(f"TEST EXAMPLE {index}")
    print("=" * 90)
    print("Source:", example.get("source", "unknown"))
    print("Paper ID:", example.get("paper_id", "unknown"))

    print("\n----- LORA MODEL OUTPUT -----\n")
    print(generated)

    print("\n----- EXPECTED OUTPUT -----\n")
    print(example["completion"])


def main():
    parser = argparse.ArgumentParser(description="Test a Hugging Face base model plus LoRA adapter on held-out review examples.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--test-file", type=Path, default=TEST_FILE)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Missing adapter directory: {args.adapter_dir}")
    if not args.test_file.exists():
        raise SystemExit(f"Missing test file: {args.test_file}")

    examples = load_examples(args.test_file, args.limit)
    tokenizer, model = load_model(args.base_model, args.adapter_dir, args.local_files_only)

    print("Base model:", args.base_model)
    print("Adapter:", args.adapter_dir)
    print("Test file:", args.test_file)
    print("Examples:", len(examples))

    for index, example in enumerate(examples, start=1):
        generated = generate_review(
            tokenizer,
            model,
            example["prompt"],
            args.max_input_tokens,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
        )
        print_result(index, example, generated)


if __name__ == "__main__":
    main()
