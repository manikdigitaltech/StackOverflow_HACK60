"""
Step 6 test: parses one of your sample PDFs and confirms literature
retrieval returns real, correctly-hydrated PeerRead matches -- not just
that FAISS returns *something*, but that titles/text/scores all check out.

Run with: python -m scripts.test_literature_retriever ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.rag.retrievers.literature_retriever import LiteratureRetriever

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_literature_retriever "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed paper: {parsed.title}")

print("\nRetrieving literature context...")
retriever = LiteratureRetriever()
context = retriever.retrieve(parsed)

print(f"\nQuery text used (truncated): {context.query_text[:200]}...")
print(f"\nTop {len(context.matches)} literature matches:")
for m in context.matches:
    print(f"\n  [score={m.similarity_score:.4f}] {m.title} ({m.year}) -- section: {m.section_type}")
    print(f"    {m.chunk_text[:150]}...")

print("\nSanity check: every match above should have a REAL paper title (not "
      "'(untitled)' or blank) and chunk_text that reads like actual paper prose. "
      "The similarity scores may be only loosely on-topic, since your test paper "
      "is likely a different domain/era than the 2017 ICLR/ACL papers in KB1 -- "
      "that's expected, not a bug.")
