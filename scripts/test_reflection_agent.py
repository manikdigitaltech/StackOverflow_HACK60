"""
Step 7 test (agent #7 of 9): Reflection Agent.

This is the biggest integration test yet -- it chains ALL SIX prior agents
(Paper Understanding, Literature RAG, Novelty, Methodology, Citation,
Evidence & Reproducibility) to produce real inputs for Reflection to
critique. Expect this to take noticeably longer than any single-agent
test so far, since it's 5 separate LLM calls before Reflection's own
(6th) call even starts. That's expected, not stuck.

Run with: python -m scripts.test_reflection_agent ".\\data\\raw_papers\\your_sample.pdf"
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
from core.agents.evidence_reproducibility_agent import EvidenceReproducibilityAgent
from core.agents.reflection_agent import ReflectionAgent

if len(sys.argv) < 2:
    print('Usage: python -m scripts.test_reflection_agent "./data/raw_papers/your_sample.pdf"')
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Parsing: {pdf_path}")
parsed = DoclingParser().parse(pdf_path)
print(f"Parsed: {parsed.title}\n")

llm = get_llm()
prompt_manager = PromptManager()

print("[1/6] Paper Understanding Agent...")
understanding = PaperUnderstandingAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[2/6] Literature RAG Agent...")
literature_context = LiteratureRAGAgent().run({"parsed_paper": parsed})

print("[3/6] Novelty Agent...")
novelty = NoveltyAgent(llm=llm, prompt_manager=prompt_manager).run({
    "paper_understanding": understanding, "literature_context": literature_context})

print("[4/6] Methodology Agent...")
methodology = MethodologyAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("[5/6] Citation Agent...")
citation = CitationAgent(llm=llm, prompt_manager=prompt_manager).run({
    "parsed_paper": parsed, "literature_context": literature_context})

print("[6/6] Evidence & Reproducibility Agent...")
evidence = EvidenceReproducibilityAgent(llm=llm, prompt_manager=prompt_manager).run(
    {"parsed_paper": parsed})

print("\nAll six upstream agents complete. Running Reflection Agent (final LLM call)...")
reflection_agent = ReflectionAgent(llm=llm, prompt_manager=prompt_manager)
result = reflection_agent.run({
    "parsed_paper": parsed,
    "novelty_assessment": novelty,
    "methodology_assessment": methodology,
    "citation_assessment": citation,
    "evidence_assessment": evidence,
})

print(f"\n--- Overall Confidence: {result.overall_confidence.upper()} ---")
print(f"--- Needs Revision: {result.needs_revision} ---")

print(f"\n--- Flags ({len(result.flags)}) ---")
if result.flags:
    for f in result.flags:
        print(f"  [{f.severity.upper()}] ({f.source_agent}) {f.flagged_item}")
        print(f"      {f.issue}")
else:
    print("  (no flags raised)")

print("\n--- Summary ---")
print(result.summary)

print("\nSanity check: does Reflection catch anything resembling the specific "
      "soft spots we already found by hand? (e.g. Novelty's speculative "
      "'fused CUDA implementation might be implied' overlap claim, or thin/"
      "generic notes anywhere). A reflection agent that raises zero flags "
      "despite known soft spots existing isn't actually doing its job.")
