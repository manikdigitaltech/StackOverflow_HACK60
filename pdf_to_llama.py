from pathlib import Path
from pypdf import PdfReader
import ollama


# -----------------------------
# 1. PDF path
# -----------------------------
pdf_path = Path("data/sample_paper.pdf")

if not pdf_path.exists():
    print("PDF not found.")
    print("Please place your PDF at: data/sample_paper.pdf")
    exit()


# -----------------------------
# 2. Extract text from PDF
# -----------------------------
reader = PdfReader(pdf_path)

paper_text = ""

# Read first 5 pages only for testing
for page_number, page in enumerate(reader.pages[:5], start=1):
    page_text = page.extract_text()

    if page_text:
        paper_text += f"\n\n--- Page {page_number} ---\n"
        paper_text += page_text


if not paper_text.strip():
    print("No text extracted from PDF.")
    print("This may be a scanned PDF. OCR may be needed later.")
    exit()


# Limit text because local model has context limit
paper_text = paper_text[:6000]


# -----------------------------
# 3. Create prompt for Llama
# -----------------------------
prompt = f"""
You are an AI Research Paper Reviewer.

Important:
RAG means Retrieval-Augmented Generation.
Do not use any other meaning of RAG.

I am giving you extracted text from a submitted research paper.

Your task:
1. Summarize the paper in simple English.
2. Identify the research problem.
3. Identify the proposed method.
4. Identify the dataset if mentioned.
5. Identify the main contribution.
6. Give a preliminary novelty opinion.

Important:
This is only a preliminary review because no external related papers
have been retrieved yet. Do not make strong novelty claims.

Extracted paper text:
{paper_text}
"""


# -----------------------------
# 4. Send to local Llama 3.2
# -----------------------------
response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 5. Print response
# -----------------------------
print("\n===== Llama 3.2 Paper Review =====\n")
print(response["message"]["content"])