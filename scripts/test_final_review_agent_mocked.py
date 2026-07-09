"""
Fast test for Final Review Generator using MOCKED upstream inputs, built
from the real outputs we've already observed from all 8 upstream agents
across this conversation's actual test runs -- not fabricated placeholder
data, but not a fresh live run either.

This is a deliberate trade-off: the full test_final_review_agent.py chains
all 9 agents (~15-30 min on CPU). Since we've already independently
verified each of the 8 upstream agents works correctly on their own, this
script isolates and tests ONLY what's new here -- Final Review's own
prompt rendering, schema validation, and the missing_baselines
deterministic override -- with a single LLM call instead of nine.

Run with: python -m scripts.test_final_review_agent_mocked
"""

from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager
from core.agents.final_review_agent import FinalReviewAgent
from core.schemas.agent_output_schemas import (
    PaperUnderstandingOutput, NoveltyAssessment, ContributionNoveltyVerdict, NoveltyComparison,
    MethodologyAssessment, MethodologyAspectVerdict,
    CitationAssessment, CitationCoverageVerdict,
    ReferenceUsageAssessment, ReferenceUsageVerdict,
    EvidenceReproducibilityAssessment, ClaimEvidenceVerdict, ReproducibilityAspectVerdict,
    FigureTableSummary, FigureSummary, TableSummary,
    VisualReferenceAssessment, VisualReferenceVerdict,
    ReflectionNotes, ReflectionFlag,
)

# --- Reconstructed from real agent outputs observed earlier in this build ---

understanding = PaperUnderstandingOutput(
    summary=("DepthWeave-KV introduces a token-adaptive cache compression method for "
             "long-context language models, achieving near-full-cache task quality while "
             "reducing KV memory usage by 8.3x and improving decode efficiency."),
    stated_contributions=[
        "Cross-depth residual factorization sharing low-rank channel bases across neighboring transformer layers",
        "A token-conditional depth router allocating higher reconstruction rank to retrieval-critical tokens",
        "Calibration-free online error tracking using attention-output probes",
    ],
    key_terms=["DepthWeave-KV", "token-adaptive caching", "cross-depth residual factorization",
               "token-conditional depth router", "attention-output probes", "KV cache compression"],
)

novelty = NoveltyAssessment(
    contribution_verdicts=[
        ContributionNoveltyVerdict(contribution="Cross-depth residual factorization", verdict="novel",
                                    note="Not mentioned in any retrieved paper."),
        ContributionNoveltyVerdict(contribution="Token-conditional depth router", verdict="partial",
                                    note="DynamicKV explores task-aware adaptive strategies, though not depth-aware."),
        ContributionNoveltyVerdict(contribution="Calibration-free online error tracking", verdict="novel",
                                    note="No retrieved paper does online attention-output probing during generation."),
    ],
    overlapping_work=[
        NoveltyComparison(
            compared_paper_title="DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs",
            similarity_note="Both address adaptive, task-aware KV cache compression."),
    ],
    novelty_rating="medium",
    reasoning=("Several contributions are genuinely novel, particularly the cross-depth "
               "factorization and online adaptation, though task-aware adaptive compression "
               "in general overlaps with DynamicKV."),
)

methodology = MethodologyAssessment(
    aspect_verdicts=[
        MethodologyAspectVerdict(aspect="baseline_comparisons", assessment="adequate",
                                  note="Compares against 9 baselines including StreamingLLM, H2O, SnapKV, MiniCache, TailorKV."),
        MethodologyAspectVerdict(aspect="ablation_studies", assessment="adequate",
                                  note="Table 2 ablates cross-depth sharing, the router, online error tracking, residual gates, and the fused kernel."),
        MethodologyAspectVerdict(aspect="hyperparameter_justification", assessment="adequate",
                                  note="Specifies rank levels rho=0/2/4/8 for different token types in the Method section."),
        MethodologyAspectVerdict(aspect="experimental_setup_clarity", assessment="adequate",
                                  note="Clearly details benchmarks (LongBench, Needle-in-a-Haystack, etc.) and evaluation protocol."),
        MethodologyAspectVerdict(aspect="statistical_rigor", assessment="weak",
                                  note="Reports single point-estimate scores with no error bars or variance across runs."),
    ],
    missing_baselines=[],
    soundness_rating="good",
    reasoning="Strong baseline coverage and thorough ablations, but lacks statistical rigor.",
)

citation = CitationAssessment(
    coverage_verdicts=[
        CitationCoverageVerdict(related_paper_title="CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios",
                                 cited=False, note="No equivalent paper found in reference list."),
        CitationCoverageVerdict(related_paper_title="DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs",
                                 cited=False, note="No equivalent paper found in reference list."),
        CitationCoverageVerdict(related_paper_title="WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference",
                                 cited=False, note="No equivalent paper found in reference list."),
        CitationCoverageVerdict(related_paper_title="KVReviver: Reversible KV Cache Compression with Sketch-Based Token Reconstruction",
                                 cited=False, note="No equivalent paper found in reference list."),
        CitationCoverageVerdict(related_paper_title="SCBench: A KV Cache-Centric Analysis of Long-Context Methods",
                                 cited=False, note="No equivalent paper found in reference list."),
    ],
    citation_quality_rating="fair",
    reasoning="All 5 retrieved, highly relevant recent papers on KV cache compression are absent from the paper's own reference list.",
)

reference_usage = ReferenceUsageAssessment(
    reference_verdicts=[
        ReferenceUsageVerdict(reference="StreamingLLM: Efficient Streaming Language Models with Attention Sinks",
                               cited_in_body=True, role="baseline", usefulness="high",
                               evidence="Table 1 reports StreamingLLM as a directly compared baseline."),
        ReferenceUsageVerdict(reference="H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs",
                               cited_in_body=True, role="baseline", usefulness="high",
                               evidence="Table 1 reports H2O as a directly compared baseline."),
        ReferenceUsageVerdict(reference="Attention Is All You Need",
                               cited_in_body=True, role="background", usefulness="low",
                               evidence="Cited once in the introduction as general transformer background, no further engagement."),
    ],
    overall_rating="good",
    summary="The paper substantively engages with its cited baselines via direct comparison, though a few background references get only a passing mention.",
)

evidence = EvidenceReproducibilityAssessment(
    claim_verdicts=[
        ClaimEvidenceVerdict(claim="DepthWeave-KV reaches 8.3x KV memory reduction and 72.8 tokens/sec at 64K context",
                              verdict="supported", note="Matches Table 1's reported 8.3x and 72.8 tokens/sec."),
    ],
    reproducibility_verdicts=[
        ReproducibilityAspectVerdict(aspect="code_availability", assessment="missing",
                                      note="No code repository or release is mentioned."),
        ReproducibilityAspectVerdict(aspect="hyperparameter_details", assessment="adequate",
                                      note="Specific rank values (rho=0,2,4,8) are given in the Method section."),
        ReproducibilityAspectVerdict(aspect="dataset_availability", assessment="missing",
                                      note="No specific training dataset or split is named."),
        ReproducibilityAspectVerdict(aspect="training_details", assessment="weak",
                                      note="Mentions a frozen teacher cache objective but few further details."),
        ReproducibilityAspectVerdict(aspect="compute_requirements", assessment="missing",
                                      note="No hardware or compute budget is specified."),
    ],
    overall_rating="good",
    reasoning="Headline claims are well-supported by Table 1, but reproducibility is limited by missing code, dataset, and compute details.",
)

figure_table = FigureTableSummary(
    figure_summaries=[
        FigureSummary(figure_id="figure_1",
                       interpretation="Shows DepthWeave-KV weaving shared depth bases across layers while preserving residual capacity for important tokens.",
                       caption_self_contained=True),
        FigureSummary(figure_id="figure_2",
                       interpretation="Depicts the architecture: shared low-rank KV channel bases, residual gates, and a token-conditional depth router.",
                       caption_self_contained=True),
    ],
    table_summaries=[
        TableSummary(table_id="table_1",
                      key_takeaway="DepthWeave-KV achieves 62.9% avg score and 8.3x memory reduction, best among compressed methods.",
                      caption_self_contained=True),
        TableSummary(table_id="table_2",
                      key_takeaway="Removing token-conditional routing or online error tracking most degrades retrieval quality; the shared-bases-only variant drops furthest (58.9%/87.6%).",
                      caption_self_contained=True),
    ],
    extraction_consistency_note="No figure/table reference mismatches detected.",
)

visual_reference = VisualReferenceAssessment(
    reference_verdicts=[
        VisualReferenceVerdict(mention="Figure 2", target_id="figure_2", exists=True,
                                purpose="method_explanation", verdict="supported",
                                evidence="Figure 2 illustrates the shared low-rank channel bases and token-conditional router.",
                                note="Prose walks through each architectural component shown in the figure."),
        VisualReferenceVerdict(mention="Table 2", target_id="table_2", exists=True,
                                purpose="ablation", verdict="supported",
                                evidence="Table 2 reports the ablation of the router and online error tracking.",
                                note="Text directly cites the ablation numbers from the table."),
    ],
    unresolved_mentions=[],
    overall_quality="good",
    summary="The paper's prose meaningfully engages with the figures/tables it references, with no dangling mentions.",
)

reflection = ReflectionNotes(
    flags=[
        ReflectionFlag(
            source_agent="novelty", flagged_item="fused CUDA implementation",
            issue="Claimed novel partly because it's absent from retrieved papers, but absence from a small corpus doesn't confirm genuine novelty.",
            severity="minor"),
    ],
    needs_revision=False,
    overall_confidence="medium",
    summary="Assessments are generally well-grounded; one minor speculative novelty claim was flagged but doesn't warrant revision.",
)

# --- Run only Final Review Agent's own LLM call ---

llm = get_llm()
prompt_manager = PromptManager()

print("Running Final Review Generator with mocked upstream inputs (1 LLM call only)...")
agent = FinalReviewAgent(llm=llm, prompt_manager=prompt_manager)
result = agent.run({
    "paper_understanding": understanding,
    "novelty_assessment": novelty,
    "methodology_assessment": methodology,
    "citation_assessment": citation,
    "reference_usage_assessment": reference_usage,
    "evidence_assessment": evidence,
    "figure_table_summary": figure_table,
    "visual_reference_assessment": visual_reference,
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

print(f"\n--- Missing Baselines (should be [] -- overridden from Methodology, which found none) ---")
print(result.missing_baselines if result.missing_baselines else "  (none)")

print(f"\n--- Final Recommendation: {result.final_recommendation.upper()} ---")
print(f"--- Confidence: {result.confidence.upper()} ---")

print("\n" + "=" * 70)
print("Sanity check: does 'weaknesses' mention missing statistical rigor and/or")
print("uncited related work? Does 'novelty_analysis' correctly temper the fused")
print("CUDA claim per Reflection's flag, rather than repeating it confidently?")
print("Did final_recommendation use underscores correctly (e.g. 'weak_accept'),")
print("confirming the duplicate-schema bug fix actually worked?")
