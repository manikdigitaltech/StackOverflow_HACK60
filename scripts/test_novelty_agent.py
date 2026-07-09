"""
Step 7 test (agent #3 of 9): Novelty Agent.

This is the first agent that COMBINES two previous agents' outputs --
Paper Understanding (contributions) and Literature RAG (retrieved
comparisons) -- so this test runs all three in sequence, mirroring
exactly how the real graph will chain them in Step 8.

Run with: python -m scripts.test_novelty_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.utils.grounding import is_title_grounded
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.paper_understanding_agent import PaperUnderstandingAgent
from core.agents.literature_rag_agent import LiteratureRAGAgent
from core.agents.novelty_agent import NoveltyAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_novelty_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}\n")

llm = get_llm()
prompt_manager = PromptManager()

print("Running Paper Understanding Agent...")
understanding_agent = PaperUnderstandingAgent(llm=llm, prompt_manager=prompt_manager)
understanding = understanding_agent.run({"parsed_paper": parsed})
print(f"  Got {len(understanding.stated_contributions)} contributions.\n")

print("Running Literature RAG Agent...")
rag_agent = LiteratureRAGAgent()
literature_context = rag_agent.run({"parsed_paper": parsed})
print(f"  Retrieved {len(literature_context.matches)} matches:")
retrieved_titles = set()
for m in literature_context.matches:
    print(f"    - {m.title}")
    retrieved_titles.add(m.title)
print()

print("Running Novelty Agent (calls the LLM, may take a moment on CPU)...")
novelty_agent = NoveltyAgent(llm=llm, prompt_manager=prompt_manager)
result = novelty_agent.run({
    "paper_understanding": understanding,
    "literature_context": literature_context,
})

print(f"\n--- Novelty Rating: {result.novelty_rating.upper()} ---")

print(f"\n--- Contribution Verdicts ({len(result.contribution_verdicts)} of "
      f"{len(understanding.stated_contributions)} contributions) ---")
for v in result.contribution_verdicts:
    print(f"  [{v.verdict.upper()}] {v.contribution}")
    print(f"      {v.note}")

print("\n--- Overlapping Work (should cite ONLY the retrieved papers above) ---")
hallucinated = []
for o in result.overlapping_work:
    grounded = is_title_grounded(o.compared_paper_title, retrieved_titles)
    match_marker = "OK" if grounded else "!! NOT IN RETRIEVED LIST !!"
    print(f'  [{match_marker}] "{o.compared_paper_title}": {o.similarity_note}')
    if not grounded:
        hallucinated.append(o.compared_paper_title)

print("\n--- Reasoning ---")
print(result.reasoning)

print("\n--- Grounding & completeness check ---")
if len(result.contribution_verdicts) < len(understanding.stated_contributions):
    print(f"NOTE: only {len(result.contribution_verdicts)} verdict(s) for "
          f"{len(understanding.stated_contributions)} stated contributions -- "
          f"the model skipped some rather than covering every one.")
else:
    print("Every stated contribution got its own verdict -- coverage is complete.")

if hallucinated:
    print(f"WARNING: {len(hallucinated)} cited title(s) do NOT match any retrieved paper -- "
          f"hallucinated citation(s):")
    for h in hallucinated:
        print(f"  - {h}")
else:
    print("All cited papers in 'overlapping_work' match the actual retrieved list. Grounding held.")
