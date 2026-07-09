import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as error:
    raise SystemExit(
        "Missing dependency: peft\n"
        "Install training dependencies first, for example:\n"
        "  .venv/bin/pip install peft accelerate\n"
    ) from error


TRAIN_FILE = Path("finetune_data/train.jsonl")
VAL_FILE = Path("finetune_data/validation.jsonl")
DEFAULT_OUTPUT_DIR = Path("models/peer_review_lora")
DEFAULT_MODEL = "distilgpt2"


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "prompt": row["prompt"].strip(),
                    "completion": row["completion"].strip(),
                    "paper_id": row.get("paper_id", ""),
                }
            )
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def format_prompt(prompt):
    return f"{prompt.strip()}\n\nReview:\n"


def tokenize_example(example, tokenizer, max_length, max_completion_length):
    prompt_text = format_prompt(example["prompt"])
    completion_text = example["completion"].strip() + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    completion_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
    )["input_ids"]

    completion_budget = min(max_completion_length, max_length - 1)
    completion_ids = completion_ids[:completion_budget]
    if not completion_ids:
        completion_ids = [tokenizer.eos_token_id]

    prompt_budget = max_length - len(completion_ids)
    prompt_ids = prompt_ids[:prompt_budget]

    input_ids = prompt_ids + completion_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + completion_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def infer_lora_targets(model):
    candidate_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
    }
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidate_suffixes:
                found.add(suffix)

    if found:
        return sorted(found)

    return ["c_attn", "c_proj"]


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a causal LM on filtered paper-review JSONL data.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base Hugging Face model name or local model path.")
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional small-run limit for testing.")
    parser.add_argument("--max-validation-rows", type=int, default=None, help="Optional small-run limit for testing.")
    parser.add_argument("--local-files-only", action="store_true", help="Load the base model/tokenizer only from the local Hugging Face cache.")
    args = parser.parse_args()

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if not args.train_file.exists():
        raise SystemExit(f"Missing training file: {args.train_file}")
    if not args.validation_file.exists():
        raise SystemExit(f"Missing validation file: {args.validation_file}")

    print("Loading datasets...")
    train_dataset = load_jsonl(args.train_file, args.max_train_rows)
    validation_dataset = load_jsonl(args.validation_file, args.max_validation_rows)
    print("Train rows:", len(train_dataset))
    print("Validation rows:", len(validation_dataset))

    print("Loading tokenizer and model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_modules = infer_lora_targets(model)
    print("LoRA target modules:", ", ".join(target_modules))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Tokenizing...")
    train_tokenized = train_dataset.map(
        lambda row: tokenize_example(row, tokenizer, args.max_length, args.max_completion_length),
        remove_columns=train_dataset.column_names,
    )
    validation_tokenized = validation_dataset.map(
        lambda row: tokenize_example(row, tokenizer, args.max_length, args.max_completion_length),
        remove_columns=validation_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=validation_tokenized,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    print("Saving LoRA adapter and tokenizer...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Training completed.")
    print("Saved adapter at:", args.output_dir)


if __name__ == "__main__":
    main()
