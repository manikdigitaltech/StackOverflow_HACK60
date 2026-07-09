"""
Shared helpers that format each agent's structured output into compact,
readable text -- used by any agent that needs to feed OTHER agents'
results into its own prompt (Reflection, Final Review), rather than
duplicating the same formatting logic in each.
"""

from core.schemas.agent_output_schemas import (
    PaperUnderstandingOutput, FigureTableSummary, NoveltyAssessment,
    MethodologyAssessment, CitationAssessment, ReferenceUsageAssessment,
    EvidenceReproducibilityAssessment, ReflectionNotes, AdversarialCritique,
    VisualReferenceAssessment,
)


def format_understanding(u: PaperUnderstandingOutput) -> str:
    lines = [f"Summary: {u.summary}", "Stated contributions:"]
    lines += [f"  - {c}" for c in u.stated_contributions]
    return "\n".join(lines)


def format_figure_table(ft: FigureTableSummary) -> str:
    lines = []
    for f in ft.figure_summaries:
        lines.append(f"  - [{f.figure_id}] {f.interpretation}")
    for t in ft.table_summaries:
        lines.append(f"  - [{t.table_id}] {t.key_takeaway}")
    if ft.extraction_consistency_note:
        lines.append(f"Consistency note: {ft.extraction_consistency_note}")
    return "\n".join(lines) if lines else "No figures or tables were extracted."


def format_visual_reference(a: VisualReferenceAssessment) -> str:
    lines = [f"Overall quality: {a.overall_quality}"]
    for v in a.reference_verdicts:
        lines.append(f"  - [{v.verdict}, {v.purpose}] {v.mention}: {v.note or v.evidence}")
    if a.unresolved_mentions:
        lines.append(f"Unresolved mentions (no matching target extracted): {', '.join(a.unresolved_mentions)}")
    lines.append(f"Summary: {a.summary}")
    return "\n".join(lines) if a.reference_verdicts or a.unresolved_mentions else a.summary


def format_novelty(a: NoveltyAssessment) -> str:
    lines = [f"Novelty rating: {a.novelty_rating}"]
    for v in a.contribution_verdicts:
        lines.append(f"  - [{v.verdict}] {v.contribution}: {v.note}")
    for o in a.overlapping_work:
        lines.append(f'  - Overlap cited: "{o.compared_paper_title}": {o.similarity_note}')
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def format_methodology(a: MethodologyAssessment) -> str:
    lines = [f"Soundness rating: {a.soundness_rating}"]
    for v in a.aspect_verdicts:
        lines.append(f"  - [{v.assessment}] {v.aspect}: {v.note}")
    if a.missing_baselines:
        lines.append(f"Missing baselines: {', '.join(a.missing_baselines)}")
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def format_citation(a: CitationAssessment) -> str:
    lines = [f"Citation quality: {a.citation_quality_rating}"]
    for v in a.coverage_verdicts:
        status = "cited" if v.cited else "NOT CITED"
        lines.append(f"  - [{status}] {v.related_paper_title}: {v.note}")
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def format_reference_usage(a: ReferenceUsageAssessment) -> str:
    lines = [f"Overall rating: {a.overall_rating}"]
    for v in a.reference_verdicts:
        status = "cited" if v.cited_in_body else "NOT CITED"
        lines.append(f"  - [{status}, {v.role}, usefulness={v.usefulness}] {v.reference}: {v.evidence}")
    lines.append(f"Summary: {a.summary}")
    return "\n".join(lines)


def format_evidence_repro(a: EvidenceReproducibilityAssessment) -> str:
    lines = [f"Overall rating: {a.overall_rating}"]
    for c in a.claim_verdicts:
        lines.append(f"  - [{c.verdict}] Claim: {c.claim} -- {c.note}")
    for v in a.reproducibility_verdicts:
        lines.append(f"  - [{v.assessment}] {v.aspect}: {v.note}")
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def format_adversarial_critique(c: AdversarialCritique) -> str:
    lines = [f"Weakest agent per the adversarial critic: {c.weakest_agent}"]
    if c.attacks:
        for a in c.attacks:
            lines.append(f"  - [{a.severity}] ({a.source_agent}) Attacked verdict: {a.attacked_verdict}")
            lines.append(f"      Counter-argument: {a.counter_argument}")
    else:
        lines.append("  (no attacks raised)")
    lines.append(f"Summary: {c.summary}")
    return "\n".join(lines)


def format_reflection(r: ReflectionNotes) -> str:
    lines = [f"Overall confidence in assessments: {r.overall_confidence}",
             f"Needs revision: {r.needs_revision}"]
    for f in r.flags:
        lines.append(f"  - [{f.severity}] ({f.source_agent}) {f.flagged_item}: {f.issue}")
    lines.append(f"Summary: {r.summary}")
    return "\n".join(lines)
