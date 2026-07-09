"""
Step 7 test (agent #5 of 9): Citation Agent.

Uses Literature RAG's retrieved matches + the paper's own parsed reference
list to check whether any highly relevant retrieved literature is missing
from the paper's actual citations. Deliberately skips Paper Understanding
Agent (not needed here -- title/abstract come straight from ParsedPaper),
saving one LLM call versus the Novelty Agent test.

Run with: python -m scripts.test_citation_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.utils.grounding import is_title_grounded
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.literature_rag_agent import LiteratureRAGAgent
from core.agents.citation_agent import CitationAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_citation_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}")
print(f"Paper has {len(parsed.references)} extracted references.\n")

llm = get_llm()
prompt_manager = PromptManager()

print("Running Literature RAG Agent...")
rag_agent = LiteratureRAGAgent()
literature_context = rag_agent.run({"parsed_paper": parsed})
retrieved_titles = set()
print(f"  Retrieved {len(literature_context.matches)} matches:")
for m in literature_context.matches:
    print(f"    - {m.title}")
    retrieved_titles.add(m.title)
print()

print("Running Citation Agent (calls the LLM, may take a moment on CPU)...")
citation_agent = CitationAgent(llm=llm, prompt_manager=prompt_manager)
result = citation_agent.run({
    "parsed_paper": parsed,
    "literature_context": literature_context,
})

print(f"\n--- Citation Quality Rating: {result.citation_quality_rating.upper()} ---")

print(f"\n--- Coverage Verdicts ({len(result.coverage_verdicts)} of "
      f"{len(literature_context.matches)} retrieved papers checked) ---")
hallucinated = []
gaps = []
for v in result.coverage_verdicts:
    grounded = is_title_grounded(v.related_paper_title, retrieved_titles)
    marker = "OK" if grounded else "!! NOT IN RETRIEVED LIST !!"
    cited_label = "CITED" if v.cited else "NOT CITED"
    print(f'  [{marker}] [{cited_label}] "{v.related_paper_title}"')
    print(f"      {v.note}")
    if not grounded:
        hallucinated.append(v.related_paper_title)
    if not v.cited:
        gaps.append(v.related_paper_title)

print("\n--- Reasoning ---")
print(result.reasoning)

print("\n--- Grounding & completeness check ---")
if len(result.coverage_verdicts) < len(literature_context.matches):
    print(f"NOTE: only {len(result.coverage_verdicts)} verdict(s) for "
          f"{len(literature_context.matches)} retrieved papers -- some were skipped "
          f"rather than every one being checked.")
else:
    print("Every retrieved paper got its own coverage verdict -- checking is complete.")

if hallucinated:
    print(f"WARNING: {len(hallucinated)} title(s) do NOT match any retrieved paper:")
    for h in hallucinated:
        print(f"  - {h}")
else:
    print("All verdict titles match the actual retrieved list. Grounding held.")

print(f"\n{len(gaps)} citation gap(s) found: {gaps if gaps else '(none)'}")
print("\nWorth a manual check: pick one 'NOT CITED' entry and look through the "
      "actual paper's reference list yourself to confirm it's genuinely missing -- "
      "that judgment call is the one thing this script can't verify automatically.")
