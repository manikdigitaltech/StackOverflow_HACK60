from pathlib import Path
from pypdf import PdfReader
import arxiv
import ollama


# -------------------------------------------------
# 1. Read submitted PDF
# -------------------------------------------------
pdf_path = Path("data/sample_paper.pdf")

if not pdf_path.exists():
    print("PDF not found.")
    print("Put your submitted paper here: data/sample_paper.pdf")
    exit()

reader = PdfReader(pdf_path)

paper_text = ""

for page_number, page in enumerate(reader.pages[:5], start=1):
    page_text = page.extract_text()
    if page_text:
        paper_text += f"\n\n--- Page {page_number} ---\n"
        paper_text += page_text

if not paper_text.strip():
    print("No text extracted from PDF.")
    exit()

# Limit text for local Llama
paper_text = paper_text[:5000]


# -------------------------------------------------
# 2. Ask Llama to create a search query
# -------------------------------------------------
query_prompt = f"""
You are helping build a RAG system for research paper novelty checking.

From the submitted paper text below, create ONE short arXiv search query.

The query should include the main topic, method, and task.
Do not write explanation. Only output the search query.

Submitted paper text:
{paper_text}
"""

query_response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": query_prompt
        }
    ]
)

search_query = query_response["message"]["content"].strip()

print("\n===== Generated Search Query =====")
print(search_query)


# -------------------------------------------------
# 3. Search related papers from arXiv
# -------------------------------------------------
search = arxiv.Search(
    query=search_query,
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)

client = arxiv.Client()

related_papers = []

print("\n===== Retrieved Related Papers from arXiv =====\n")

for i, result in enumerate(client.results(search), start=1):
    paper_info = {
        "title": result.title,
        "authors": ", ".join(author.name for author in result.authors),
        "published": str(result.published.date()),
        "url": result.entry_id,
        "summary": result.summary
    }

    related_papers.append(paper_info)

    print(f"Paper {i}")
    print("Title:", paper_info["title"])
    print("Published:", paper_info["published"])
    print("URL:", paper_info["url"])
    print("-" * 80)


if not related_papers:
    print("No related papers found from arXiv.")
    exit()


# -------------------------------------------------
# 4. Prepare retrieved evidence
# -------------------------------------------------
retrieved_text = ""

for i, paper in enumerate(related_papers, start=1):
    retrieved_text += f"""
Related Paper {i}
Title: {paper['title']}
Authors: {paper['authors']}
Published: {paper['published']}
URL: {paper['url']}
Abstract: {paper['summary']}
"""


# -------------------------------------------------
# 5. Send submitted paper + retrieved papers to Llama
# -------------------------------------------------
review_prompt = f"""
You are a Novelty Review Agent for scientific papers.

Important:
RAG means Retrieval-Augmented Generation.
Do not use any other meaning of RAG.

Your task:
Compare the submitted paper with the retrieved related papers.

Novelty output must be exactly one of:
1. High Novelty
2. Moderate Novelty
3. Low Novelty
4. Similar Work Already Exists

Compare these points:
- Research problem
- Proposed method
- Dataset
- Baselines
- Results
- Main contribution

Evidence rule:
Use only the submitted paper text and retrieved related papers.
Do not guess.
If evidence is missing, say "Not enough evidence".

Submitted paper text:
{paper_text}

Retrieved related papers:
{retrieved_text}

Give final output in this format:

Novelty Level:
Reason:
Similar Papers Found:
What Already Exists:
Possible New Contribution:
Missing Evidence:
Final Reviewer Comment:
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": review_prompt
        }
    ]
)

print("\n===== RAG-Based Novelty Review =====\n")
print(response["message"]["content"])
