from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


MODEL_DIR = "models/peer_review_model"


SECTION_HEADERS = [
    "Summary:",
    "Strengths:",
    "Weaknesses:",
    "Novelty:",
    "Recommendation:",
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()

    return tokenizer, model


def clean_generated_text(text):
    """
    Remove repeated section headings and unnecessary generated text.
    """

    # Normalize strange bullet symbols
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove repeated section headings
    for header in SECTION_HEADERS:
        text = re.sub(rf"\b{re.escape(header)}", "", text, flags=re.IGNORECASE)

    # Remove common bad fragments
    bad_phrases = [
        "Accepted:",
        "Rejected:",
        "(the manuscript)",
        "### Review:",
        "### Paper:",
        "### Instruction:",
    ]

    for phrase in bad_phrases:
        text = text.replace(phrase, "")

    # Remove extra spaces and blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def cut_at_next_section(text):
    """
    If the model starts generating another section, stop there.
    """

    positions = []

    for header in SECTION_HEADERS:
        match = re.search(rf"\b{re.escape(header)}", text, flags=re.IGNORECASE)
        if match:
            positions.append(match.start())

    if positions:
        text = text[:min(positions)]

    return text.strip()


def generate_text(tokenizer, model, prompt, max_new_tokens=80):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.35,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    generated_text = cut_at_next_section(generated_text)
    generated_text = clean_generated_text(generated_text)

    return generated_text


def clean_bullets(text, fallback_bullets):
    """
    Convert generated text into clean bullet points.
    """

    text = clean_generated_text(text)

    lines = text.splitlines()
    bullets = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = line.lstrip("-").strip()

        if len(line) < 8:
            continue

        bullets.append(line)

    # If model generated one paragraph, split into sentences
    if not bullets:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                bullets.append(sentence)

    # Use fallback if output is still bad
    if not bullets:
        bullets = fallback_bullets

    # Keep only first 3 bullets
    bullets = bullets[:3]

    return "\n".join([f"- {bullet}" for bullet in bullets])


def clean_paragraph(text, fallback):
    text = clean_generated_text(text)

    # Remove bullet markers from paragraph sections
    text = text.replace("- ", " ")

    # Keep only first 2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if sentences:
        return " ".join(sentences[:2])

    return fallback


def clean_recommendation(text):
    text_lower = text.lower()

    if "accept" in text_lower and "reject" not in text_lower:
        return "Accept"

    if "reject" in text_lower:
        return "Reject"

    return "Reject"


def generate_review(title, abstract):
    tokenizer, model = load_model()

    base_prompt = f"""### Instruction:
Write a scientific peer review for the given research paper.
Use reviewer-style academic language.
Do not repeat section headings inside the answer.

### Paper:
Title:
{title}

Abstract:
{abstract}

Paper Text:

"""

    summary_prompt = base_prompt + """### Review:
Write only the Summary section. Do not write Strengths, Weaknesses, Novelty, or Recommendation.

Summary:
"""

    strengths_prompt = base_prompt + """### Review:
Write only the Strengths section as bullet points. Do not write any other section.

Strengths:
-"""

    weaknesses_prompt = base_prompt + """### Review:
Write only the Weaknesses section as bullet points. Do not write any other section.

Weaknesses:
-"""

    novelty_prompt = base_prompt + """### Review:
Write only the Novelty section. Do not write any other section.

Novelty:
"""

    recommendation_prompt = base_prompt + """### Review:
Choose only one word: Accept or Reject.

Recommendation:
"""

    summary_raw = generate_text(
        tokenizer,
        model,
        summary_prompt,
        max_new_tokens=90
    )

    strengths_raw = generate_text(
        tokenizer,
        model,
        strengths_prompt,
        max_new_tokens=80
    )

    weaknesses_raw = generate_text(
        tokenizer,
        model,
        weaknesses_prompt,
        max_new_tokens=80
    )

    novelty_raw = generate_text(
        tokenizer,
        model,
        novelty_prompt,
        max_new_tokens=70
    )

    recommendation_raw = generate_text(
        tokenizer,
        model,
        recommendation_prompt,
        max_new_tokens=10
    )

    summary = clean_paragraph(
        summary_raw,
        fallback="The paper presents a deep learning approach for medical image classification and evaluates it against baseline methods."
    )

    strengths = clean_bullets(
        strengths_raw,
        fallback_bullets=[
            "The paper addresses a relevant medical image classification problem.",
            "The proposed approach is evaluated against baseline methods.",
            "The work attempts to improve classification accuracy using deep learning."
        ]
    )

    weaknesses = clean_bullets(
        weaknesses_raw,
        fallback_bullets=[
            "The paper would benefit from stronger experimental validation.",
            "The comparison with competitive baselines should be clearer.",
            "The limitations and failure cases need more detailed discussion."
        ]
    )

    novelty = clean_paragraph(
        novelty_raw,
        fallback="The novelty appears moderate because the work applies deep learning to a known medical image classification task."
    )

    recommendation = clean_recommendation(recommendation_raw)

    final_review = f"""Summary:
{summary}

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Novelty:
{novelty}

Recommendation:
{recommendation}"""

    return final_review


if __name__ == "__main__":
    title = "A Deep Learning Approach for Medical Image Classification"

    abstract = (
        "This paper proposes a convolutional neural network model for classifying "
        "medical images into multiple disease categories. The method is evaluated "
        "on a public dataset and compared with standard machine learning baselines. "
        "The results show improved accuracy over traditional feature-based methods."
    )

    review = generate_review(title, abstract)

    print("\nGenerated Review:")
    print("=" * 80)
    print(review)