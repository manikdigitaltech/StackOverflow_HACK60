import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


MODEL_NAME = "distilgpt2"

TRAIN_FILE = Path("finetune_data/train.jsonl")
VAL_FILE = Path("finetune_data/validation.jsonl")

OUTPUT_DIR = "models/peer_review_model"


def load_jsonl(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            text = row["prompt"] + row["completion"]
            rows.append({"text": text})

    return Dataset.from_list(rows)


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )


def main():
    print("Loading dataset...")

    train_dataset = load_jsonl(TRAIN_FILE)
    val_dataset = load_jsonl(VAL_FILE)

    print("Train rows:", len(train_dataset))
    print("Validation rows:", len(val_dataset))

    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id

    print("Tokenizing data...")

    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed successfully.")
    print("Model saved at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()