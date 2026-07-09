import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_DATASETS = [
    "JerMa88/ICLR_Peer_Reviews",
    "Vidushee/openreview-peer-reviews",
]

OUTPUT_DIR = Path("finetune_data")
RAW_DIR = Path("hf_openreview_raw")
SOURCE_NAME = "hf_openreview_filtered"
RANDOM_SEED = 42

MIN_REVIEW_CHARS = 500
MIN_CONTEXT_CHARS = 200
MAX_INPUT_CHARS = 12000
MAX_COMPLETION_CHARS = 7000

TITLE_KEYS = ["title", "paper_title"]
ABSTRACT_KEYS = ["abstract"]
TEXT_KEYS = ["full_text", "paper_text", "text"]
REVIEW_KEYS = ["review"]
RATING_KEYS = ["rating", "overall_score"]
CONFIDENCE_KEYS = ["confidence", "experience_assessment"]
YEAR_KEYS = ["year"]
PAPER_ID_KEYS = ["paper_id", "forum", "review_id"]

REQUIRED_SECTIONS = ["Summary:", "Strengths:", "Weaknesses:", "Novelty:", "Recommendation:"]

SYNTHETIC_OR_PROMPT_PATTERNS = [
    r"\blet me review this paper\b",
    r"\bi need to provide\b",
    r"\bi'?ll review this\b",
    r"\bas an ai\b",
    r"\byou are a member of the scientific community\b",
    r"###\s*paper content",
    r"\bthinking trace\b",
]

PROCESS_NOISE_PATTERNS = [
    r"\bdear authors\b",
    r"\bcommittee final decision\b",
    r"\bauthor response\b",
    r"\brebuttal\b",
    r"\bofficial comment\b",
    r"\bpost rebuttal\b",
]

TEMPLATE_PATTERNS = [
    r"write a scientific peer review",
    r"reviewer-style academic language",
    r"the paper addresses a relevant and meaningful research problem",
    r"the proposed work attempts to contribute",
    r"no detailed reviewer comments are available",
]


def normalize_whitespace(value):
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_missing(value):
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "null", "n/a", "na", "[]", "{}"}


def first_value(row, keys):
    for key in keys:
        if key in row and not is_missing(row[key]):
            return normalize_whitespace(row[key])
    return ""


def has_pattern(text, patterns):
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def download_parquet_files(dataset_name, cache_dir):
    files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    parquet_files = [file for file in files if file.endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for {dataset_name}")

    local_paths = []
    for file in parquet_files:
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=file,
            cache_dir=cache_dir,
        )
        local_paths.append(Path(local_path))
    return local_paths


def load_dataset_rows(dataset_name, cache_dir):
    rows = []
    for path in download_parquet_files(dataset_name, cache_dir):
        frame = pd.read_parquet(path)
        for row in frame.to_dict(orient="records"):
            row["_hf_dataset"] = dataset_name
            row["_hf_file"] = path.name
            rows.append(row)
    return rows


def split_review_points(review):
    paragraphs = [
        normalize_whitespace(part)
        for part in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", review)
    ]
    paragraphs = [part for part in paragraphs if len(part) >= 60]

    summary = []
    strengths = []
    weaknesses = []
    novelty = []

    for paragraph in paragraphs:
        lower = paragraph.lower()
        if any(word in lower for word in ["strength", "strong", "clear", "well-written", "interesting"]):
            strengths.append(paragraph)
        if any(word in lower for word in ["weakness", "concern", "unclear", "missing", "lack", "limited", "not clear"]):
            weaknesses.append(paragraph)
        if any(word in lower for word in ["novel", "original", "incremental", "contribution", "prior work"]):
            novelty.append(paragraph)
        if not summary and any(word in lower for word in ["paper", "work", "authors", "method", "proposes", "present"]):
            summary.append(paragraph)

    if not summary:
        summary = paragraphs[:2]
    if not strengths:
        strengths = [p for p in paragraphs if p not in weaknesses][:3]
    if not weaknesses:
        weaknesses = paragraphs[1:4]
    if not novelty:
        novelty = paragraphs[:1]

    return summary[:2], strengths[:3], weaknesses[:3], novelty[:1]


def normalize_recommendation(row):
    rating = first_value(row, RATING_KEYS)
    if not rating:
        return "Not specified"
    lower = rating.lower()
    if "reject" in lower:
        return "Reject"
    if "accept" in lower:
        return "Accept"
    score_match = re.search(r"-?\d+(?:\.\d+)?", rating)
    if score_match:
        score = float(score_match.group())
        if score >= 6:
            return "Accept"
        if score <= 3:
            return "Reject"
    return rating[:120]


def build_prompt(row, allow_title_only=False):
    title = first_value(row, TITLE_KEYS)
    abstract = first_value(row, ABSTRACT_KEYS)
    full_text = first_value(row, TEXT_KEYS)

    context_chars = len(abstract) + len(full_text)
    if not title:
        return "", "missing_title"
    if not allow_title_only and context_chars < MIN_CONTEXT_CHARS:
        return "", "insufficient_paper_context"

    parts = [
        "You are reviewing a machine learning research submission. Provide a rigorous, constructive review based only on the paper information below.",
        "",
        "Title:",
        title,
    ]
    if abstract:
        parts.extend(["", "Abstract:", abstract])
    if full_text:
        parts.extend(["", "Paper Text:", full_text[:MAX_INPUT_CHARS]])
    parts.extend(["", "Required Output Sections:", "Summary, Strengths, Weaknesses, Novelty, Recommendation"])
    return normalize_whitespace("\n".join(parts)), ""


def build_completion(row):
    review = first_value(row, REVIEW_KEYS)
    summary, strengths, weaknesses, novelty = split_review_points(review)
    recommendation = normalize_recommendation(row)
    confidence = first_value(row, CONFIDENCE_KEYS)

    parts = [
        "Summary:",
        "\n\n".join(summary),
        "",
        "Strengths:",
        "\n".join(f"- {item}" for item in strengths),
        "",
        "Weaknesses:",
        "\n".join(f"- {item}" for item in weaknesses),
        "",
        "Novelty:",
        "\n\n".join(novelty),
        "",
        "Recommendation:",
        recommendation,
    ]
    if confidence:
        parts.extend(["", "Reviewer Confidence:", confidence[:120]])
    return normalize_whitespace("\n".join(parts))[:MAX_COMPLETION_CHARS]


def quality_reject_reason(row, allow_title_only=False):
    review = first_value(row, REVIEW_KEYS)
    prompt, prompt_reason = build_prompt(row, allow_title_only=allow_title_only)
    combined = "\n\n".join([prompt, review])

    if prompt_reason:
        return prompt_reason
    if len(review) < MIN_REVIEW_CHARS:
        return "short_review"
    if has_pattern(review, SYNTHETIC_OR_PROMPT_PATTERNS):
        return "synthetic_or_prompt_like_review"
    if has_pattern(combined, PROCESS_NOISE_PATTERNS):
        return "review_process_noise"
    if has_pattern(combined, TEMPLATE_PATTERNS):
        return "template_text"
    if len(set(re.findall(r"[A-Za-z]{4,}", review.lower()))) < 80:
        return "low_information_review"
    return ""


def make_paper_id(row, index):
    paper_id = first_value(row, PAPER_ID_KEYS)
    if paper_id:
        return paper_id
    title = first_value(row, TITLE_KEYS).lower()
    title = re.sub(r"[^a-z0-9]+", "_", title).strip("_")[:80]
    return title or f"hf_openreview_{index:06d}"


def make_example(row, index, allow_title_only=False):
    reason = quality_reject_reason(row, allow_title_only=allow_title_only)
    if reason:
        return None, reason

    prompt, _ = build_prompt(row, allow_title_only=allow_title_only)
    completion = build_completion(row)
    if not all(section in completion for section in REQUIRED_SECTIONS):
        return None, "missing_required_section"

    year = first_value(row, YEAR_KEYS)
    if str(year).isdigit():
        year = int(year)

    return {
        "source": SOURCE_NAME,
        "original_split": first_value(row, ["_hf_dataset"]),
        "paper_id": make_paper_id(row, index),
        "year": year,
        "prompt": prompt,
        "completion": completion,
    }, ""


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face OpenReview peer-review datasets into filtered fine-tuning JSONL.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--allow-title-only", action="store_true", help="Allow rows that have a title and review but no abstract/full text.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick testing.")
    args = parser.parse_args()

    all_rows = []
    for dataset_name in args.datasets:
        print(f"Loading {dataset_name}...")
        rows = load_dataset_rows(dataset_name, args.raw_dir)
        print(f"Loaded {len(rows)} rows from {dataset_name}")
        all_rows.extend(rows)

    examples = []
    reject_reasons = Counter()
    seen = set()

    for index, row in enumerate(all_rows, start=1):
        example, reason = make_example(row, index, allow_title_only=args.allow_title_only)
        if reason:
            reject_reasons[reason] += 1
            continue
        dedupe_key = (example["paper_id"], example["completion"][:300])
        if dedupe_key in seen:
            reject_reasons["duplicate"] += 1
            continue
        seen.add(dedupe_key)
        examples.append(example)
        if args.max_rows and len(examples) >= args.max_rows:
            break

    if not examples:
        print("Reject reasons:", dict(reject_reasons))
        raise SystemExit("No examples passed quality filtering.")

    random.seed(args.seed)
    random.shuffle(examples)

    train_end = max(1, int(len(examples) * 0.85))
    validation_end = max(train_end, int(len(examples) * 0.925))
    train_rows = examples[:train_end]
    validation_rows = examples[train_end:validation_end]
    test_rows = examples[validation_end:]

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "validation.jsonl", validation_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)

    print("Rows loaded:", len(all_rows))
    print("Rows kept:", len(examples))
    print("Rows rejected:", sum(reject_reasons.values()))
    print("Reject reasons:", dict(reject_reasons))
    print("Train rows:", len(train_rows))
    print("Validation rows:", len(validation_rows))
    print("Test rows:", len(test_rows))
    print("Output folder:", args.output_dir)


if __name__ == "__main__":
    main()
