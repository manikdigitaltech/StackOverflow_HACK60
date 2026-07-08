from pathlib import Path
import json
import re


# --------------------------------------------------
# Paths
# --------------------------------------------------

PEERREAD_ROOT = Path("PeerRead/data/arxiv.cs.cl_2007-2017")
OUTPUT_ROOT = Path("peerread_processed")

SPLITS = ["train", "dev", "test"]


# --------------------------------------------------
# Safe text helper
# --------------------------------------------------

def safe_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


# --------------------------------------------------
# Read JSON safely
# --------------------------------------------------

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None


# --------------------------------------------------
# Extract paper text
# --------------------------------------------------

def extract_paper_text(parsed_pdf_json):
    metadata = parsed_pdf_json.get("metadata", {})

    title = safe_text(metadata.get("title", ""))
    abstract = safe_text(metadata.get("abstractText", ""))

    paper_text = ""

    sections = parsed_pdf_json.get("sections", [])

    for section in sections:
        heading = safe_text(section.get("heading", ""))
        text = safe_text(section.get("text", ""))

        if heading or text:
            paper_text += f"\n\n{heading}\n{text}"

    # Limit paper body length for practical fine-tuning
    paper_text = paper_text[:8000]

    final_text = f"""Title:
{title}

Abstract:
{abstract}

Paper Text:
{paper_text}
"""

    return final_text.strip(), title, abstract


# --------------------------------------------------
# Extract useful review comments only
# --------------------------------------------------

def extract_review_text(review_json):
    """
    Extract only useful human review text.
    Avoids copying metadata JSON such as conference, VERSION, title, abstract, etc.
    """

    review_parts = []

    useful_keys = [
        "comments",
        "review",
        "summary",
        "strengths",
        "weaknesses",
        "evaluation",
        "decision"
    ]

    def collect_from_dict(obj):
        if not isinstance(obj, dict):
            return

        for key in useful_keys:
            value = obj.get(key)

            if isinstance(value, str) and value.strip():
                review_parts.append(f"{key}:\n{value.strip()}")

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        review_parts.append(item.strip())
                    elif isinstance(item, dict):
                        collect_from_dict(item)

            elif isinstance(value, dict):
                collect_from_dict(value)

        # Handle nested reviews
        reviews_value = obj.get("reviews")
        if isinstance(reviews_value, list):
            for item in reviews_value:
                if isinstance(item, dict):
                    collect_from_dict(item)
                elif isinstance(item, str) and item.strip():
                    review_parts.append(item.strip())

        elif isinstance(reviews_value, dict):
            collect_from_dict(reviews_value)

    if isinstance(review_json, dict):
        collect_from_dict(review_json)

    elif isinstance(review_json, list):
        for item in review_json:
            if isinstance(item, dict):
                collect_from_dict(item)
            elif isinstance(item, str) and item.strip():
                review_parts.append(item.strip())

    if not review_parts:
        return "No detailed reviewer comments are available."

    return "\n\n".join(review_parts).strip()


# --------------------------------------------------
# Extract recommendation from review JSON and text
# --------------------------------------------------

def get_recommendation(review_json, review_text):
    """
    Extract recommendation from text or numerical score.
    """

    combined_text = ""

    if isinstance(review_json, dict):
        combined_text += json.dumps(review_json, ensure_ascii=False).lower()

    combined_text += "\n" + safe_text(review_text).lower()

    # Text labels
    if "strong accept" in combined_text:
        return "Strong Accept"

    if "weak accept" in combined_text:
        return "Weak Accept"

    if "strong reject" in combined_text:
        return "Strong Reject"

    if "weak reject" in combined_text:
        return "Weak Reject"

    if "accept" in combined_text and "reject" not in combined_text:
        return "Accept"

    if "reject" in combined_text:
        return "Reject"

    # Numerical recommendation score
    numbers = re.findall(r"recommendation[^0-9]*(\d+)", combined_text)

    if numbers:
        score = int(numbers[-1])

        if score >= 7:
            return "Accept"
        elif score >= 5:
            return "Weak Accept"
        else:
            return "Reject"

    return "Reject"


# --------------------------------------------------
# Shorten text safely
# --------------------------------------------------

def short_text(text, max_words=55):
    text = safe_text(text)
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


# --------------------------------------------------
# Build paper-specific review output
# --------------------------------------------------

def make_review_style_output(title, abstract, review_text, review_json):
    title = safe_text(title)
    abstract = safe_text(abstract)
    review_text = safe_text(review_text)

    recommendation = get_recommendation(review_json, review_text)

    abstract_short = short_text(abstract, max_words=65)

    if not title:
        title = "the submitted paper"

    if not abstract_short:
        abstract_short = "the research problem described in the manuscript"

    output = f"""Summary:
The paper titled "{title}" presents a research contribution on {abstract_short}. The work is evaluated as a scientific submission based on the manuscript content and reviewer comments.

Strengths:
- The paper addresses a relevant and meaningful research problem.
- The proposed work attempts to contribute to the existing literature.
- The method appears to target an identifiable limitation in prior work.
- The manuscript presents a concrete approach that can be evaluated experimentally.

Weaknesses:
- The paper would benefit from stronger experimental evidence.
- The comparison with competitive baseline methods should be clearer and more systematic.
- Some claims require deeper justification and stronger empirical support.
- The discussion of limitations and failure cases could be improved.

Novelty:
The contribution appears moderate. The idea is connected to an important research direction, but its novelty and impact should be supported through stronger validation, clearer positioning against prior work, and more convincing empirical analysis.

Recommendation:
{recommendation}"""

    return output.strip()


# --------------------------------------------------
# Convert one split
# --------------------------------------------------

def convert_split(split):
    print(f"\nProcessing split: {split}")

    parsed_dir = PEERREAD_ROOT / split / "parsed_pdfs"
    reviews_dir = PEERREAD_ROOT / split / "reviews"

    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{split}_review_style.jsonl"

    if not parsed_dir.exists():
        print(f"Missing parsed_pdfs folder: {parsed_dir}")
        return

    if not reviews_dir.exists():
        print(f"Missing reviews folder: {reviews_dir}")
        return

    parsed_files = list(parsed_dir.glob("*.pdf.json"))

    count = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as out:

        for parsed_file in parsed_files:
            paper_id = parsed_file.name.replace(".pdf.json", "")
            review_file = reviews_dir / f"{paper_id}.json"

            if not review_file.exists():
                skipped += 1
                continue

            parsed_json = load_json(parsed_file)
            review_json = load_json(review_file)

            if parsed_json is None or review_json is None:
                skipped += 1
                continue

            paper_text, title, abstract = extract_paper_text(parsed_json)
            review_text = extract_review_text(review_json)

            if not paper_text or not review_text:
                skipped += 1
                continue

            review_style_output = make_review_style_output(
                title=title,
                abstract=abstract,
                review_text=review_text,
                review_json=review_json
            )

            example = {
                "instruction": (
                    "Write a scientific peer review for the given research paper. "
                    "Include Summary, Strengths, Weaknesses, Novelty, and Recommendation. "
                    "Use reviewer-style academic language."
                ),
                "input": paper_text,
                "output": review_style_output
            }

            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved: {output_file}")
    print(f"Examples created: {count}")
    print(f"Skipped files: {skipped}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    if not PEERREAD_ROOT.exists():
        print("PeerRead dataset path not found:")
        print(PEERREAD_ROOT)
        exit()

    for split in SPLITS:
        convert_split(split)

    print("\nDone.")
    print("Created review-style JSONL files inside:")
    print(OUTPUT_ROOT)