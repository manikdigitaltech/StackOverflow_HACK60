"""
Step 7 test (agent #10 of 10 -- the last one): Final Review Generator.

This is the BIGGEST integration test in the whole project so far: it
chains all 9 prior calls (Paper Understanding, Literature RAG, Novelty,
Methodology, Citation, Reference Usage, Evidence & Reproducibility,
Figure & Table, Reflection) before Final Review's own (10th) call. 9 real
LLM calls total, sequentially, on CPU. Budget real time for this --
potentially well over 10-15 minutes given how long some individual calls
(Citation, Evidence & Reproducibility, Reflection) have taken on their
own. Progress prints after each step so you can see it moving.

Run with: python -m scripts.test_final_review_agent ".\\data\\raw_papers\\your_sample.pdf"
"""

import sys
from core.parsing.docling_parser import DoclingParser
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.paper_understanding_agent import PaperUnderstandingAgent
from core.agents.literature_rag_agent import LiteratureRAGAgent
from core.agents.novelty_agent import NoveltyAgent
from core.agents.methodology_agent import MethodologyAgent
from core.agents.citation_agent import CitationAgent
from core.agents.reference_usage_agent import ReferenceUsageAgent
from core.agents.evidence_reproducibility_agent import EvidenceReproducibilityAgent
from core.agents.figure_table_agent import FigureTableAgent
from core.agents.reflection_agent import ReflectionAgent
from core.agents.final_review_agent import FinalReviewAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_final_review_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}\n")

llm = get_llm()
prompt_manager = PromptManager()

print("[1/10] Paper Understanding Agent...")
understanding = PaperUnderstandingAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[2/10] Literature RAG Agent...")
literature_context = LiteratureRAGAgent().run({"parsed_paper": parsed})

print("[3/10] Novelty Agent...")
novelty = NoveltyAgent(llm=llm, prompt_manager=prompt_manager).run({
    "paper_understanding": understanding, "literature_context": literature_context})

print("[4/10] Methodology Agent...")
methodology = MethodologyAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[5/10] Citation Agent...")
citation = CitationAgent(llm=llm, prompt_manager=prompt_manager).run({
    "parsed_paper": parsed, "literature_context": literature_context})

print("[6/10] Reference Usage Agent...")
reference_usage = ReferenceUsageAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[7/10] Evidence & Reproducibility Agent...")
evidence = EvidenceReproducibilityAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[8/10] Figure & Table Agent...")
figure_table = FigureTableAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[9/10] Reflection Agent...")
reflection = ReflectionAgent(llm=llm, prompt_manager=prompt_manager).run({
    "parsed_paper": parsed,
    "novelty_assessment": novelty,
    "methodology_assessment": methodology,
    "citation_assessment": citation,
    "evidence_assessment": evidence,
})

print("\nAll 9 upstream agents complete. Running Final Review Generator (10th and final call)...")
final_review_agent = FinalReviewAgent(llm=llm, prompt_manager=prompt_manager)
result = final_review_agent.run({
    "paper_understanding": understanding,
    "figure_table_summary": figure_table,
    "novelty_assessment": novelty,
    "methodology_assessment": methodology,
    "citation_assessment": citation,
    "reference_usage_assessment": reference_usage,
    "evidence_assessment": evidence,
    "reflection_notes": reflection,
})

print("\n" + "=" * 70)
print("FINAL REVIEW")
print("=" * 70)

print(f"\n--- Paper Summary ---\n{result.paper_summary}")

print(f"\n--- Strengths ({len(result.strengths)}) ---")
for i, s in enumerate(result.strengths, 1):
    print(f"  {i}. {s}")

print(f"\n--- Weaknesses ({len(result.weaknesses)}) ---")
for i, w in enumerate(result.weaknesses, 1):
    print(f"  {i}. {w}")

print(f"\n--- Questions for Authors ({len(result.questions_for_authors)}) ---")
for i, q in enumerate(result.questions_for_authors, 1):
    print(f"  {i}. {q}")

print(f"\n--- Novelty Analysis ---\n{result.novelty_analysis}")
print(f"\n--- Citation Quality ---\n{result.citation_quality}")
print(f"\n--- Reference Usage Quality ---\n{result.reference_usage_quality}")
print(f"\n--- Reproducibility ---\n{result.reproducibility}")
print(f"\n--- Evidence Mapping ---\n{result.evidence_mapping}")

print(f"\n--- Missing Baselines (copied directly from Methodology Agent) ---")
if result.missing_baselines:
    for b in result.missing_baselines:
        print(f"  - {b}")
else:
    print("  (none)")

print(f"\n--- Final Recommendation: {result.final_recommendation.upper()} ---")
print(f"--- Confidence: {result.confidence.upper()} ---")

print("\n" + "=" * 70)
print("Sanity check: does this review actually reflect everything we found by hand")
print("across all 8 upstream agents? Specifically check: does 'weaknesses' mention")
print("the missing statistical rigor / uncited related work? Do 'questions_for_authors'")
print("ask something SPECIFIC (e.g. 'what base model was used?'), not generic filler?")
print("Does missing_baselines match what Methodology Agent found on its own?")
