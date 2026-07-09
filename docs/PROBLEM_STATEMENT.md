# Autonomous AI Paper Reviewer & Scientific Evaluation Agent

> Build an agentic AI system that reads a research paper end-to-end, evaluates it like a
> peer reviewer, grounds every judgment in retrieved literature, and produces a structured,
> explainable review report — all within a 24 GB GPU budget.

---

## 1. Objective

- Analyze research papers, including **text, figures, tables, and references**.
- Evaluate **novelty, methodology, technical soundness, and experimental quality**.
- Compare contributions with existing literature using **Retrieval-Augmented Generation (RAG)**.
- Detect **unsupported claims, missing experiments, and reproducibility issues**.
- Generate **structured, explainable, and evidence-backed** peer-review reports.
- Leverage **Agentic AI, Large Language Models (LLMs), Vision-Language Models (VLMs), and Deep Learning**.
- Improve review quality through a **self-reflection (verifier)** step.
- Support **Human-in-the-Loop** approval before the final recommendation.

---

## 2. Requirements

- Understand the complete research paper, including figures and tables.
- Evaluate novelty by comparing with existing literature.
- Analyze methodology, datasets, experiments, and evaluation metrics.
- Detect missing baselines, weak experimental evidence, and unsupported claims.
- Assess reproducibility and citation quality.
- Coordinate multiple specialized agents to generate a unified review.
- Request human approval before issuing the final recommendation.

---

## 3. Challenges

- Understanding long scientific documents.
- Grounding reviews with relevant literature while minimizing hallucinations.
- Coordinating multiple AI agents to produce consistent decisions.
- Running the complete pipeline within the provided **24 GB GPU** budget.

---

## 4. Techniques

- Multi-Agent AI Workflow
- Retrieval-Augmented Generation (RAG)
- Scientific Document Understanding
- Vision-Language Models for Figures & Tables
- Transformer-based Deep Learning Models
- Citation & Similarity Analysis
- Self-Reflection and Verifier Agents
- Human-in-the-Loop Review
- Explainable AI (XAI)
- Knowledge Graph-based Reasoning

---

## 5. Tools & Frameworks

*Participants may choose alternative frameworks, libraries & models as per their preference.*

| Layer | Suggested options |
|---|---|
| Deep learning | PyTorch / TensorFlow |
| Vision / image processing | OpenCV |
| Agent orchestration | LangGraph / CrewAI |
| LLMs | Llama 3.1 8B or Llama 3.2 3B; Qwen2.5 7B/14B (4-bit quantized for 14B) |
| Vector store & chains | FAISS / LangChain |
| PDF parsing | PyMuPDF / Docling |
| UI | Streamlit |
| Evaluation | DeepEval / RAGAS |

---

## 6. Technical Architecture

```
                      Research Paper (PDF)
              Text • Figures • Tables • References
                              |
        ┌─────────────── Scientific Document Understanding ───────────────┐
        │ PDF Parsing · Layout Detection · OCR · Figure Extraction ·      │
        │ Table Extraction · Reference Parsing                            │
        └────────────────────────────────────────────────────────────────┘
                              |
             Multi-Agent Controller (LangGraph / CrewAI)
                              |
   ┌──────────────┬───────────────────┬──────────────────┬─────────────────┐
   │ Novelty Agent│   Method Agent    │  Evidence Agent  │ Citation Agent  │
   │  Literature  │   Experiments,    │ Missing Baselines│   Similarity    │
   │  Comparison  │   Methodology     │                  │   References    │
   └──────────────┴───────────────────┴──────────────────┴─────────────────┘
                              |
   ┌────────── RAG Layer ───────────┐     ┌──── Scientific Reasoning LLM ────┐
   │ PeerRead Dataset               │ ──▶ │ Llama 3.1 / Qwen2.5 / Transformers│
   │ Semantic Scholar API           │     └──────────────────────────────────┘
   │ arXiv Literature               │                    |
   │   ↓                            │     ┌── Self-Reflection & Verifier Agent ──┐
   │ FAISS + LangChain KB           │     │ Consistency Check · Hallucination    │
   └────────────────────────────────┘     │ Detection · Evidence Validation ·    │
                                          │ Explainability                       │
                                          └──────────────────────────────────────┘
                                                         |
                                            Human-in-the-Loop Approval
                                                         |
   ┌──────────────────────── Final Review Report ────────────────────────┐
   │ Paper Summary · Strengths · Weaknesses · Novelty · Reproducibility  │
   │ Rating · Reviewer Confidence · Recommendation                       │
   └────────────────────────────────────────────────────────────────────┘
```

### Pipeline stages

1. **Scientific Document Understanding** — PDF parsing, layout detection, OCR, figure extraction, table extraction, reference parsing.
2. **Multi-Agent Controller** (LangGraph / CrewAI) — dispatches to specialized agents.
3. **Specialized Agents** — Novelty (literature comparison), Method (experiments, methodology), Evidence (missing baselines), Citation (similarity, references).
4. **RAG Layer** — PeerRead dataset, Semantic Scholar API, arXiv literature → FAISS + LangChain knowledge base.
5. **Scientific Reasoning LLM** — Llama 3.1 / Qwen2.5 / transformer models.
6. **Self-Reflection & Verifier Agent** — consistency check, hallucination detection, evidence validation, explainability.
7. **Human-in-the-Loop Approval**.
8. **Final Review Report**.

---

## 7. Final Output

The system must produce:

- Paper Summary
- Strengths
- Weaknesses
- Questions for Authors / Rebuttal
- Final Review Conclusion
- **Review Rating** — *Accept / Weak Accept / Borderline / Weak Reject / Reject*
- Reviewer Confidence
- Rating Justification

### Optional Bonus

- Novelty Analysis
- Citation Quality Assessment
- Reproducibility Assessment
- Explainable Evidence Mapping
- Missing Experiments & Baselines

---

## 8. Brownie Points

- **Quantitative agreement vs. PeerRead ground truth** — accuracy / F1 / Cohen's κ on the test split.
- **Rebuttal-aware re-review** — the agent revises its verdict after a simulated author rebuttal.
- **Full local deployment** — the entire checklist above running locally.

---

## 9. Datasets

| Status | Dataset | Purpose | Link |
|---|---|---|---|
| **Required (Fixed Evaluation Set)** | PeerRead (ICLR 2017 Subset) | Official benchmark containing research papers, expert peer reviews, and accept/reject labels. **All teams must use this dataset** for training and evaluation to ensure fair comparison. | https://github.com/allenai/PeerRead |
| Optional (Retrieval Only) | Semantic Scholar API | Retrieve related literature for **RAG-based novelty verification**, citation grounding, and supporting evidence. | https://www.semanticscholar.org/product/api |
| Optional (Retrieval Only) | arXiv | Retrieve recent research papers for **literature search**, related work comparison, and novelty assessment. | https://arxiv.org |

### Data split rules

> Use the repository's built-in **80/10/10 train/dev/test split (~1.3K official reviews)** — **do not create your own split.**
> Fine-tune and develop **only on the train set**; report all metrics on the **test set**.
> The two optional sources may be used freely for retrieval but are **not** part of the fixed evaluation set, and using them (or not) has **no bearing on grading**.

---

## 10. Evaluation

- **Dataset:** PeerRead (ICLR 2017)
- **Retrieval:** Semantic Scholar, arXiv
- **Hardware:** 24 GB GPU
- **Metrics:**
  - Precision, Recall, F1 Score
  - Review Accuracy
  - Hallucination Rate
  - Evidence Grounding
  - Reviewer Agreement
  - Novelty Detection Accuracy

---

## 11. Constraints Summary

| Constraint | Value |
|---|---|
| GPU budget | 24 GB (full pipeline must fit) |
| Evaluation dataset | PeerRead ICLR 2017 subset (mandatory) |
| Split | Repository's built-in 80/10/10 (no custom splits) |
| Training data | Train set only |
| Reported metrics | Test set only |
| Model size guidance | ≤ 8B full precision, or 14B with 4-bit quantization |
| Final decision | Requires human-in-the-loop approval |