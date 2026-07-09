import json
import random
import re
from pathlib import Path


PEERREAD_DATA_ROOT = Path("PeerRead/data")
OUTPUT_DIR = Path("finetune_data")

BEST_SUBSETS = ["iclr_2017", "acl_2017"]
OPTIONAL_SMALL_SUBSETS = ["conll_2016"]
SPLITS = ["train", "dev", "test"]

RANDOM_SEED = 42
MAX_INPUT_CHARS = 12000
MAX_REVIEW_CHARS = 9000


INSTRUCTION = (
    "Write a scientific peer review for the given research paper. "
    "Include Summary, Strengths, Weaknesses, Novelty, and Recommendation. "
    "Use specific evidence from the paper and reviewer-style academic language."
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_whitespace(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_paper_input(parsed_json, review_json):
    metadata = parsed_json.get("metadata", {})

    title = safe_text(metadata.get("title")) or safe_text(review_json.get("title"))
    abstract = safe_text(metadata.get("abstractText")) or safe_text(review_json.get("abstract"))

    sections = []
    for section in parsed_json.get("sections", []):
        heading = safe_text(section.get("heading"))
        text = safe_text(section.get("text"))
        if heading or text:
            sections.append(f"{heading}\n{text}".strip())

    paper_text = normalize_whitespace("\n\n".join(sections))
    paper_text = paper_text[:MAX_INPUT_CHARS]

    return normalize_whitespace(
        f"""Title:
{title}

Abstract:
{abstract}

Paper Text:
{paper_text}"""
    )


def recommendation_from_review(review_json, reviews):
    accepted = review_json.get("accepted")
    if accepted is True:
        return "Accept"
    if accepted is False:
        return "Reject"

    scores = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        for key in ["RECOMMENDATION", "recommendation"]:
            value = review.get(key)
            if value is None:
                continue
            match = re.search(r"\d+", str(value))
            if match:
                scores.append(int(match.group()))

    if scores:
        return "Accept" if sum(scores) / len(scores) >= 5 else "Reject"

    return "Not available"


def extract_review_comments(review_json):
    comments = []
    seen = set()

    for review in review_json.get("reviews", []):
        if isinstance(review, str):
            comment = safe_text(review)
            key = normalize_whitespace(comment).lower()
            if comment and key not in seen:
                seen.add(key)
                comments.append(comment)
            continue

        if not isinstance(review, dict):
            continue

        title = safe_text(review.get("TITLE") or review.get("title"))
        comment = safe_text(review.get("comments") or review.get("review"))

        parts = []
        if title:
            parts.append(title)
        if comment:
            parts.append(comment)

        if parts:
            comment = normalize_whitespace("\n".join(parts))
            key = comment.lower()
            if key not in seen:
                seen.add(key)
                comments.append(comment)

    return [normalize_whitespace(comment) for comment in comments if len(comment) > 40]


def split_review_evidence(comments):
    combined = "\n\n".join(comments)

    strengths = []
    weaknesses = []
    general = []

    for paragraph in re.split(r"\n\s*\n", combined):
        paragraph = normalize_whitespace(paragraph)
        if len(paragraph) < 40:
            continue

        lower = paragraph.lower()
        if any(word in lower for word in ["strength", "pros", "positive", "interesting"]):
            strengths.append(paragraph)
        elif any(word in lower for word in ["weakness", "cons", "concern", "lack", "unclear", "not clear", "insufficient"]):
            weaknesses.append(paragraph)
        else:
            general.append(paragraph)

    return general, strengths, weaknesses


def first_items(items, count, fallback):
    cleaned = []
    seen = set()
    for item in items:
        item = re.sub(r"^(strengths?|weaknesses?|pros|cons)\s*:?", "", item, flags=re.IGNORECASE).strip(" -")
        key = normalize_whitespace(item).lower()
        if len(item) > 30 and key not in seen:
            seen.add(key)
            cleaned.append(item[:700])

    if not cleaned:
        cleaned = fallback

    return cleaned[:count]


def build_review_output(review_json, comments):
    title = safe_text(review_json.get("title")) or "the submitted paper"
    abstract = safe_text(review_json.get("abstract"))
    recommendation = recommendation_from_review(review_json, review_json.get("reviews", []))

    general, strengths_raw, weaknesses_raw = split_review_evidence(comments)

    summary_source = abstract or (general[0] if general else comments[0])
    summary_source = summary_source[:900]

    strengths = first_items(
        strengths_raw or general,
        3,
        ["The paper addresses a relevant research problem and presents a concrete technical approach."],
    )
    weaknesses = first_items(
        weaknesses_raw or general[1:],
        3,
        ["The paper would benefit from clearer experimental evidence, stronger comparisons, and deeper discussion of limitations."],
    )

    novelty_source = first_items(
        general + strengths_raw + weaknesses_raw,
        1,
        ["The novelty should be judged against prior work and the strength of the empirical validation."],
    )[0]

    output = f"""Summary:
The paper titled "{title}" studies {summary_source}

Strengths:
{chr(10).join(f"- {item}" for item in strengths)}

Weaknesses:
{chr(10).join(f"- {item}" for item in weaknesses)}

Novelty:
{novelty_source}

Recommendation:
{recommendation}"""

    return normalize_whitespace(output[:MAX_REVIEW_CHARS])


def make_prompt(paper_input):
    return f"""### Instruction:
{INSTRUCTION}

### Paper:
{paper_input}

### Review:
"""


def iter_examples(include_conll=False):
    subsets = BEST_SUBSETS + (OPTIONAL_SMALL_SUBSETS if include_conll else [])

    for subset in subsets:
        for split in SPLITS:
            split_dir = PEERREAD_DATA_ROOT / subset / split
            parsed_dir = split_dir / "parsed_pdfs"
            reviews_dir = split_dir / "reviews"

            if not parsed_dir.exists() or not reviews_dir.exists():
                continue

            for parsed_file in sorted(parsed_dir.glob("*.pdf.json")):
                paper_id = parsed_file.name.replace(".pdf.json", "")
                review_file = reviews_dir / f"{paper_id}.json"
                if not review_file.exists():
                    continue

                parsed_json = load_json(parsed_file)
                review_json = load_json(review_file)
                comments = extract_review_comments(review_json)
                if not comments:
                    continue

                paper_input = extract_paper_input(parsed_json, review_json)
                completion = build_review_output(review_json, comments)

                yield {
                    "source": subset,
                    "original_split": split,
                    "paper_id": paper_id,
                    "prompt": make_prompt(paper_input),
                    "completion": completion,
                }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    raise SystemExit(
        "This generator is disabled because it creates PeerRead review-dataset "
        "examples from iclr_2017/acl_2017. Use a new approved source instead."
    )

    random.seed(RANDOM_SEED)

    if not PEERREAD_DATA_ROOT.exists():
        raise SystemExit("PeerRead/data not found. Clone PeerRead before preparing data.")

    rows = list(iter_examples(include_conll=False))
    random.shuffle(rows)

    train_end = int(len(rows) * 0.85)
    validation_end = int(len(rows) * 0.925)

    train_rows = rows[:train_end]
    validation_rows = rows[train_end:validation_end]
    test_rows = rows[validation_end:]

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUTPUT_DIR / "validation.jsonl", validation_rows)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_rows)

    print("Prepared Llama fine-tuning data from real PeerRead reviews.")
    print("Selected subsets:", ", ".join(BEST_SUBSETS))
    print("Skipped arXiv subsets because their review files contain no human reviews.")
    print("Train rows:", len(train_rows))
    print("Validation rows:", len(validation_rows))
    print("Test rows:", len(test_rows))
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
