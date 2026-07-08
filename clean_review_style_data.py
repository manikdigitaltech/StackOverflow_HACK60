import json
import re
from pathlib import Path


# ----------------------------------------------------
# Files to clean
# ----------------------------------------------------
FILES = [
    (
        Path("peerread_processed/train/train_review_style.jsonl"),
        Path("peerread_processed/train/train_review_style_clean.jsonl")
    ),
    (
        Path("peerread_processed/dev/dev_review_style.jsonl"),
        Path("peerread_processed/dev/dev_review_style_clean.jsonl")
    ),
    (
        Path("peerread_processed/test/test_review_style.jsonl"),
        Path("peerread_processed/test/test_review_style_clean.jsonl")
    ),
]


# ----------------------------------------------------
# Clean output function
# ----------------------------------------------------
def clean_output(old_output):
    """
    Removes copied reviewer evidence and keeps only:
    Summary, Strengths, Weaknesses, Novelty, Recommendation.
    """

    recommendation = "Reject"

    rec_match = re.search(
        r"Recommendation\s*:\s*(Strong Accept|Weak Accept|Accept|Strong Reject|Weak Reject|Reject)",
        old_output,
        flags=re.IGNORECASE
    )

    if rec_match:
        recommendation = rec_match.group(1).strip()

    # Remove everything after Human Review Evidence
    clean_text = re.split(
        r"\n\s*Human Review Evidence\s*:",
        old_output,
        flags=re.IGNORECASE
    )[0].strip()

    # Remove unwanted reviewer metadata
    bad_patterns = [
        r"Reviewer\s*\d+\s*:",
        r"IS_META_REVIEW\s*:.*",
        r"REVIEWER_CONFIDENCE\s*:.*",
        r"OTHER_KEYS\s*:.*",
        r"DATE\s*:.*",
        r"TITLE\s*:.*",
        r"comments\s*:",
        r"Pros\s*:",
        r"Cons\s*:",
    ]

    for pattern in bad_patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)

    # Add required sections if missing
    if "Summary:" not in clean_text:
        clean_text = "Summary:\n" + clean_text

    if "Strengths:" not in clean_text:
        clean_text += (
            "\n\nStrengths:\n"
            "The paper addresses an interesting and relevant research problem. "
            "The proposed method is presented clearly and attempts to solve an important limitation in the existing literature."
        )

    if "Weaknesses:" not in clean_text:
        clean_text += (
            "\n\nWeaknesses:\n"
            "The paper requires stronger experimental evidence and clearer comparisons with strong baseline methods. "
            "Some claims need better justification through more systematic evaluation."
        )

    if "Novelty:" not in clean_text:
        clean_text += (
            "\n\nNovelty:\n"
            "The contribution appears moderate. The idea is useful, but the paper needs stronger validation to clearly establish its novelty and impact."
        )

    # Remove any old recommendation section and add clean one
    clean_text = re.sub(
        r"\n*Recommendation\s*:\s*.*",
        "",
        clean_text,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    clean_text += f"\n\nRecommendation:\n{recommendation}"

    # Remove extra blank lines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

    return clean_text


# ----------------------------------------------------
# Main cleaning process
# ----------------------------------------------------
def main():
    for input_file, output_file in FILES:

        if not input_file.exists():
            print("Input file not found:")
            print(input_file)
            print("-" * 60)
            continue

        cleaned_count = 0

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:

            for line in infile:
                if not line.strip():
                    continue

                row = json.loads(line)

                row["instruction"] = (
                    "Write a scientific peer review for the given research paper. "
                    "Include Summary, Strengths, Weaknesses, Novelty, and Recommendation. "
                    "Use reviewer-style academic language."
                )

                row["output"] = clean_output(row.get("output", ""))

                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                cleaned_count += 1

        print("Cleaning completed successfully.")
        print("Input file:", input_file)
        print("Output file:", output_file)
        print("Total cleaned rows:", cleaned_count)
        print("-" * 60)


if __name__ == "__main__":
    main()