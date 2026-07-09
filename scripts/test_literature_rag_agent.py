"""
Step 7 test (agent #2 of 9): Literature RAG Agent.

This agent makes NO LLM call -- it's a thin wrapper around the retriever
already proven working in Step 6. This test mainly confirms the agent
interface wraps it correctly, not new retrieval behavior.

Run with: python -m scripts.test_literature_rag_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.agents.literature_rag_agent import LiteratureRAGAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_literature_rag_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}\n")

agent = LiteratureRAGAgent()
context = agent.run({"parsed_paper": parsed})

print(f"\nQuery used: {context.query_text[:150]}...")
print(f"\nTop {len(context.matches)} matches:")
for m in context.matches:
    print(f"  [{m.similarity_score:.4f}] {m.title} ({m.year})")

print("\nSanity check: these results should match Step 6's retrieval test "
      "exactly (same underlying retriever, just called through the agent "
      "interface now) -- confirming the wrapper adds no behavior change.")
