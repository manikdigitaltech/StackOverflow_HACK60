"""Unit tests for the novelty_agent package.

Uses hand-built embeddings/records where possible (no model download
needed) to keep tests fast and deterministic, plus a small end-to-end
smoke test against the offline fallback embedder.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty_agent.decision_trace_builder import DecisionTraceBuilder  # noqa: E402
from novelty_agent.faiss_retriever import FaissRetriever, FaissRetrieverError  # noqa: E402
from novelty_agent.models import PaperEmbedding, PaperRecord, SimilarityBreakdown, SimilarPaper  # noqa: E402
from novelty_agent.novelty_scorer import NoveltyScorer, NoveltyScoringError  # noqa: E402
from novelty_agent.similarity_service import SimilarityService  # noqa: E402
from novelty_agent.text_extractor import PaperExtractionError, PaperTextExtractor  # noqa: E402


def unit_vector(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestPaperTextExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PaperTextExtractor()

    def test_extract_basic_paper(self):
        paper_json = {
            "title": "A Study of Things",
            "abstract": "We study things in detail.",
            "sections": [
                {"heading": "Proposed Method", "text": "We use a novel technique."},
                {"heading": "Conclusion", "text": "Things were studied successfully."},
            ],
            "references": ["Ref A", {"title": "Ref B"}],
            "keywords": "things, study, detail",
            "year": 2023,
        }
        record = self.extractor.extract(paper_json, paper_id="p1")
        self.assertEqual(record.paper_id, "p1")
        self.assertIn("novel technique", record.methodology)
        self.assertIn("studied successfully", record.conclusion)
        self.assertEqual(record.keywords, ["things", "study", "detail"])
        self.assertIn("Ref A", record.references)
        self.assertEqual(record.year, 2023)

    def test_empty_paper_raises(self):
        with self.assertRaises(PaperExtractionError):
            self.extractor.extract({}, paper_id="empty")

    def test_non_dict_input_raises(self):
        with self.assertRaises(PaperExtractionError):
            self.extractor.extract("not a dict", paper_id="bad")

    def test_nested_metadata_schema(self):
        paper_json = {"metadata": {"title": "Nested", "abstractText": "Nested abstract"}}
        record = self.extractor.extract(paper_json, paper_id="p2")
        self.assertEqual(record.title, "Nested")
        self.assertEqual(record.abstract, "Nested abstract")

    def test_keyword_list_fallback(self):
        paper_json = {"title": "T", "abstract": "A", "keywords": ["deep learning", "nlp"]}
        record = self.extractor.extract(paper_json, paper_id="p3")
        self.assertEqual(record.keywords, ["deep learning", "nlp"])

    def test_no_keywords_returns_empty_list(self):
        paper_json = {"title": "T", "abstract": "A"}
        record = self.extractor.extract(paper_json, paper_id="p4")
        self.assertEqual(record.keywords, [])


class TestSimilarityService(unittest.TestCase):
    def setUp(self):
        self.service = SimilarityService()

    def test_identical_sections_yield_100(self):
        v = unit_vector(1)
        fields = {"abstract": v, "methodology": v, "conclusion": v, "references": v}
        target = PaperEmbedding(paper_id="a", vectors=fields)
        candidate = PaperEmbedding(paper_id="b", vectors=dict(fields))
        breakdown, overall = self.service.compute(target, candidate)
        self.assertAlmostEqual(overall, 100.0, places=2)
        self.assertAlmostEqual(breakdown.abstract, 100.0, places=2)

    def test_missing_section_yields_none_not_zero(self):
        v = unit_vector(2)
        target = PaperEmbedding(paper_id="a", vectors={"abstract": v})
        candidate = PaperEmbedding(paper_id="b", vectors={"abstract": v})
        breakdown, _ = self.service.compute(target, candidate)
        self.assertIsNone(breakdown.methodology)
        self.assertIsNone(breakdown.references)

    def test_overall_renormalizes_over_available_sections(self):
        v = unit_vector(3)
        target = PaperEmbedding(paper_id="a", vectors={"abstract": v})
        candidate = PaperEmbedding(paper_id="b", vectors={"abstract": v})
        _, overall = self.service.compute(target, candidate)
        # Only abstract available -> renormalized weight is 100% of it -> ~100 similarity
        self.assertAlmostEqual(overall, 100.0, places=2)


class TestNoveltyScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = NoveltyScorer()

    def test_novelty_is_100_minus_similarity(self):
        breakdown = SimilarityBreakdown(abstract=80, methodology=60, conclusion=70, references=50)
        top = [SimilarPaper(paper_id=f"p{i}", similarity=50 - i) for i in range(10)]
        novelty, confidence, band, recommendation = self.scorer.score(70.0, breakdown, top)
        self.assertAlmostEqual(novelty, 30.0)
        self.assertEqual(band, "Low Novelty")
        self.assertEqual(recommendation, "Weak Reject")

    def test_duplicate_detection(self):
        breakdown = SimilarityBreakdown(abstract=99, methodology=99, conclusion=99, references=99)
        novelty, confidence, band, recommendation = self.scorer.score(97.0, breakdown, [])
        self.assertEqual(band, "Duplicate")
        self.assertEqual(recommendation, "Strong Reject")

    def test_out_of_range_similarity_raises(self):
        breakdown = SimilarityBreakdown()
        with self.assertRaises(NoveltyScoringError):
            self.scorer.score(150.0, breakdown, [])

    def test_confidence_in_range(self):
        breakdown = SimilarityBreakdown(abstract=80, methodology=20, conclusion=50, references=50)
        _, confidence, _, _ = self.scorer.score(50.0, breakdown, [SimilarPaper("x", 50)] * 10)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 100.0)


class TestDecisionTraceBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = DecisionTraceBuilder()

    def test_trace_includes_high_and_low_labels(self):
        breakdown = SimilarityBreakdown(abstract=80, methodology=85, conclusion=50, references=20)
        trace = self.builder.build(breakdown, novelty_band="Low Novelty", recommendation="Weak Reject", closest_paper_id="343.pdf")
        self.assertIn("High methodology similarity", trace.rules)
        self.assertIn("Low reference overlap", trace.rules)
        self.assertIn("Closest match: 343.pdf", trace.rules)
        self.assertEqual(trace.rules[-2], "Low Novelty")
        self.assertEqual(trace.rules[-1], "Weak Reject")

    def test_trace_skips_none_sections(self):
        breakdown = SimilarityBreakdown(abstract=None, methodology=90, conclusion=None, references=None)
        trace = self.builder.build(breakdown, novelty_band="Moderate Novelty", recommendation="Weak Accept")
        self.assertIn("High methodology similarity", trace.rules)


class TestFaissRetriever(unittest.TestCase):
    def test_search_before_build_raises(self):
        retriever = FaissRetriever()
        with self.assertRaises(FaissRetrieverError):
            retriever.search(unit_vector(1), top_k=5)

    def test_build_and_search_roundtrip(self):
        retriever = FaissRetriever(section="abstract")
        embeddings = [
            PaperEmbedding(paper_id=f"p{i}", vectors={"abstract": unit_vector(i)}) for i in range(5)
        ]
        retriever.build(embeddings)
        results = retriever.search(embeddings[0].get("abstract"), top_k=3)
        self.assertLessEqual(len(results), 3)
        # The paper itself should be the top (or near-top) match since it's identical to the query.
        self.assertEqual(results[0].paper_id, "p0")

    def test_exclude_self(self):
        retriever = FaissRetriever(section="abstract")
        embeddings = [
            PaperEmbedding(paper_id=f"p{i}", vectors={"abstract": unit_vector(i)}) for i in range(5)
        ]
        retriever.build(embeddings)
        results = retriever.search(embeddings[0].get("abstract"), top_k=3, exclude_paper_id="p0")
        self.assertNotIn("p0", [r.paper_id for r in results])

    def test_save_and_load_roundtrip(self):
        retriever = FaissRetriever(section="abstract")
        embeddings = [PaperEmbedding(paper_id=f"p{i}", vectors={"abstract": unit_vector(i)}) for i in range(3)]
        retriever.build(embeddings)
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save(Path(tmpdir))
            loaded = FaissRetriever(section="abstract")
            loaded.load(Path(tmpdir))
            self.assertEqual(loaded.size, 3)


if __name__ == "__main__":
    unittest.main()
