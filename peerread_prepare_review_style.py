from pathlib import Path
import json


# --------------------------------------------------
# Paths
# --------------------------------------------------
PEERREAD_DATA_ROOT = Path("PeerRead/data")
OUTPUT_ROOT = Path("peerread_processed")

SPLITS = ["train", "dev", "test"]


# --------------------------------------------------
# Read JSON safely
# --------------------------------------------------
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


# --------------------------------------------------
# Extract paper input
# --------------------------------------------------
def extract_paper_text(parsed_json):
    metadata = parsed_json.get("metadata", {})

    title = metadata.get("title", "")
    abstract = metadata.get("abstractText", "")

    sections = parsed_json.get("sections", [])
    paper_text = ""

    for section in sections[:5]:  # first few sections only
        heading = section.get("heading", "")
        text = section.get("text", "")

        if heading or text:
            paper_text += f"\n\n{heading}\n{text}"

    final_input = f"""
Title:
{title}

Abstract:
{abstract}

Paper Text:
{paper_text}
"""

    return final_input.strip()


# --------------------------------------------------
# Extract real human reviews
# --------------------------------------------------
def extract_human_reviews(review_json):
    reviews = review_json.get("reviews", [])

    if not reviews:
        return ""

    review_texts = []

    for idx, review in enumerate(reviews, start=1):
        if isinstance(review, str):
            review_texts.append(f"Reviewer {idx}:\n{review}")

        elif isinstance(review, dict):
            parts = []

            for key, value in review.items():
                if isinstance(value, str) and value.strip():
                    parts.append(f"{key}:\n{value.strip()}")

                elif isinstance(value, (int, float, bool)):
                    parts.append(f"{key}: {value}")

            if parts:
                review_texts.append(f"Reviewer {idx}:\n" + "\n\n".join(parts))

    return "\n\n".join(review_texts).strip()


# --------------------------------------------------
# Convert raw review into clean reviewer-style output
# --------------------------------------------------
def build_review_style_output(review_json, human_review_text):
    accepted = review_json.get("accepted", None)

    if accepted is True:
        recommendation = "Accept"
    elif accepted is False:
        recommendation = "Reject"
    else:
        recommendation = "Not available"

    title = review_json.get("title", "")
    abstract = review_json.get("abstract", "")

    output = f"""
Summary:
The paper titled "{title}" presents the following work:
{abstract}

Human Review Evidence:
{human_review_text}

Recommendation:
{recommendation}
"""

    return output.strip()


# --------------------------------------------------
# Convert one split
# --------------------------------------------------
def convert_split(split):
    print(f"\nProcessing split: {split}")

    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{split}_review_style.jsonl"

    total_pairs = 0
    kept = 0
    skipped_empty_reviews = 0
    skipped_missing_pair = 0

    # Find all official split folders:
    # Example:
    # PeerRead/data/arxiv.cs.cl_2007-2017/train/
    split_dirs = [p for p in PEERREAD_DATA_ROOT.rglob(split) if p.is_dir()]

    with open(output_file, "w", encoding="utf-8") as out:
        for split_dir in split_dirs:
            parsed_dir = split_dir / "parsed_pdfs"
            reviews_dir = split_dir / "reviews"

            if not parsed_dir.exists() or not reviews_dir.exists():
                continue

            parsed_files = list(parsed_dir.glob("*.pdf.json"))

            for parsed_file in parsed_files:
                total_pairs += 1

                paper_id = parsed_file.name.replace(".pdf.json", "")
                review_file = reviews_dir / f"{paper_id}.json"

                if not review_file.exists():
                    skipped_missing_pair += 1
                    continue

                parsed_json = load_json(parsed_file)
                review_json = load_json(review_file)

                if parsed_json is None or review_json is None:
                    skipped_missing_pair += 1
                    continue

                human_review_text = extract_human_reviews(review_json)

                if not human_review_text:
                    skipped_empty_reviews += 1
                    continue

                paper_input = extract_paper_text(parsed_json)
                paper_input = paper_input[:8000]

                review_output = build_review_style_output(
                    review_json,
                    human_review_text
                )

                example = {
                    "instruction": (
                        "Write a scientific peer review for the given research paper. "
                        "Include Summary, Strengths, Weaknesses, Novelty, and Recommendation. "
                        "Use reviewer-style academic language."
                    ),
                    "input": paper_input,
                    "output": review_output
                }

                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Saved: {output_file}")
    print(f"Total paper-review pairs checked: {total_pairs}")
    print(f"Useful examples kept: {kept}")
    print(f"Skipped empty reviews: {skipped_empty_reviews}")
    print(f"Skipped missing pairs: {skipped_missing_pair}")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    if not PEERREAD_DATA_ROOT.exists():
        print("PeerRead/data folder not found.")
        exit()

    for split in SPLITS:
        convert_split(split)

    print("\nDone.")
    print("Clean review-style JSONL files created inside peerread_processed/")