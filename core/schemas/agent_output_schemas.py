"""
Pydantic models for parsed paper content. These are what docling_parser.py
produces and what every downstream agent consumes.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel


class Reference(BaseModel):
    raw_text: str
    title: Optional[str] = None
    year: Optional[int] = None


class Section(BaseModel):
    name: str                          # canonical name if recognized (Abstract, Method, ...), else raw heading text
    raw_heading: Optional[str] = None
    text: str
    page_start: Optional[int] = None


class Table(BaseModel):
    table_id: str
    page: Optional[int] = None
    markdown: str
    caption: Optional[str] = None


class Figure(BaseModel):
    figure_id: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None   # [l, t, r, b] in Docling's native PDF coordinate space
    image_path: Optional[str] = None     # populated by figure_cropper.py, not by docling_parser.py
    caption: Optional[str] = None
    ocr_text: Optional[str] = None       # reserved for the vision-optional extension


class ParsedPaper(BaseModel):
    title: str
    abstract: str
    sections: List[Section]
    tables: List[Table]
    figures: List[Figure]
    references: List[Reference]
    source_pdf_path: str


class LiteratureMatch(BaseModel):
    paper_id: str   # kanishka's LiteratureIndex uses venue-prefixed string ids (e.g. "iclr_2017:304"), not ints
    title: str
    year: Optional[int] = None
    chunk_text: str
    section_type: Optional[str] = None
    similarity_score: float


class LiteratureContext(BaseModel):
    query_text: str
    matches: List[LiteratureMatch]


class PaperUnderstandingOutput(BaseModel):
    summary: str                          # 2-4 sentence high-level summary of the paper
    stated_contributions: List[str]       # what the authors explicitly claim as their contributions
    key_terms: List[str]                  # important technical terms/concepts, useful context for other agents


class ContributionNoveltyVerdict(BaseModel):
    contribution: str                              # restates one of the paper's stated contributions
    verdict: Literal["novel", "overlaps", "partial"]
    note: str                                       # grounded justification; if overlaps/partial, must name the retrieved paper


class NoveltyComparison(BaseModel):
    compared_paper_title: str    # MUST exactly match a title from the retrieved LiteratureContext
    similarity_note: str         # how this specific paper overlaps or differs


class NoveltyAssessment(BaseModel):
    contribution_verdicts: List[ContributionNoveltyVerdict]   # one entry per stated contribution -- forces per-item judgment
    overlapping_work: List[NoveltyComparison]
    novelty_rating: Literal["low", "medium", "high"]
    reasoning: str


class MethodologyAspectVerdict(BaseModel):
    aspect: Literal[
        "baseline_comparisons", "ablation_studies",
        "hyperparameter_justification", "experimental_setup_clarity",
        "statistical_rigor",
    ]
    assessment: Literal["adequate", "weak", "missing"]
    note: str   # must cite specific evidence from the paper (a table, a named baseline, a specific ablation result, etc.)


class MethodologyAssessment(BaseModel):
    aspect_verdicts: List[MethodologyAspectVerdict]   # exactly 5 -- one per fixed aspect above
    missing_baselines: List[str]                       # specific methods that seem like natural baselines but weren't included
    soundness_rating: Literal["poor", "fair", "good", "excellent"]
    reasoning: str


class CitationCoverageVerdict(BaseModel):
    related_paper_title: str   # MUST exactly match a title from the retrieved LiteratureContext
    cited: bool                 # is this paper (or something clearly equivalent) present in the paper's own reference list?
    note: str                   # brief justification


class CitationAssessment(BaseModel):
    coverage_verdicts: List[CitationCoverageVerdict]   # one entry per retrieved paper -- forces per-item checking
    citation_quality_rating: Literal["poor", "fair", "good", "excellent"]
    reasoning: str


class ClaimEvidenceVerdict(BaseModel):
    claim: str    # a specific quantitative claim pulled from the paper's abstract/intro
    verdict: Literal["supported", "unsupported", "partially_supported"]
    note: str     # must cite the specific table/number that supports it, or note that none was found


class ReproducibilityAspectVerdict(BaseModel):
    aspect: Literal[
        "code_availability", "hyperparameter_details", "dataset_availability",
        "training_details", "compute_requirements",
    ]
    assessment: Literal["adequate", "weak", "missing"]
    note: str


class EvidenceReproducibilityAssessment(BaseModel):
    claim_verdicts: List[ClaimEvidenceVerdict]                     # 3-5 headline quantitative claims, checked against tables
    reproducibility_verdicts: List[ReproducibilityAspectVerdict]   # exactly 5 -- one per fixed aspect above
    overall_rating: Literal["poor", "fair", "good", "excellent"]
    reasoning: str


class ReflectionFlag(BaseModel):
    source_agent: Literal["novelty", "methodology", "citation", "evidence_reproducibility"]
    flagged_item: str    # which specific verdict/claim is being questioned
    issue: str            # what's wrong: speculative, unsupported, inconsistent, etc.
    severity: Literal["minor", "moderate", "major"]


class ReflectionNotes(BaseModel):
    flags: List[ReflectionFlag]
    needs_revision: bool                                # true only if at least one "major" flag exists
    overall_confidence: Literal["low", "medium", "high"]
    summary: str


class AdversarialAttack(BaseModel):
    # Deliberately excludes "novelty" -- the adversarial critic is scoped to
    # only Methodology, Citation, and Evidence & Reproducibility.
    source_agent: Literal["methodology", "citation", "evidence_reproducibility"]
    attacked_verdict: str    # the EXACT verdict/note/rating being attacked, quoted or closely
                              # paraphrased from that agent's own output -- not a vague summary
    counter_argument: str    # a concrete rebuttal engaging with the specific evidence cited,
                              # not a restatement of the assessment's own reasoning
    severity: Literal["minor", "moderate", "major"]


class AdversarialCritique(BaseModel):
    attacks: List[AdversarialAttack]
    weakest_agent: Literal["methodology", "citation", "evidence_reproducibility"]
    summary: str


class FigureSummary(BaseModel):
    figure_id: str
    interpretation: str            # what this figure appears to show, grounded in caption text only (no vision)
    caption_self_contained: bool   # can a reader understand it from the caption alone?


class TableSummary(BaseModel):
    table_id: str
    key_takeaway: str               # the main finding, citing specific numbers from the table's actual data
    caption_self_contained: bool


class FigureTableSummary(BaseModel):
    figure_summaries: List[FigureSummary]
    table_summaries: List[TableSummary]
    extraction_consistency_note: str = ""   # set deterministically by the agent, not by the LLM


class FinalReview(BaseModel):
    paper_summary: str
    strengths: List[str]
    weaknesses: List[str]
    questions_for_authors: List[str]
    novelty_analysis: str
    citation_quality: str
    reproducibility: str
    evidence_mapping: str
    missing_baselines: List[str] = []   # set deterministically from MethodologyAssessment, not re-derived by the LLM
    final_recommendation: Literal["reject", "weak_reject", "borderline", "weak_accept", "accept"]
    confidence: Literal["low", "medium", "high"]
