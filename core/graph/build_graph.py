"""
Assembles the review orchestration graph. Wires the 9 review agents into one
bounded, checkpointed LangGraph run, then gates the final recommendation behind
a mandatory human-in-the-loop approval interrupt (DB persistence is still a
separate Phase 2 concern).

Shape:
    START --> paper_understanding --\\
          \\-> literature_rag --------+--> novelty ----------\\
          \\-> figure_table (also -\\  \\-> citation           \\
          \\-> methodology         |  \\------------------------> reflection --[revise]--> prepare_revision --\\
          \\-> evidence_repro -----/------------------------------/    |                                       |
                                                                        [proceed]                     (loops back to
                                                                        v                               novelty/methodology/
                                                              ready_for_synthesis <-- figure_table       citation/evidence_repro)
                                                                        |
                                                                        v
                                                                  final_review --> human_approval --> END
                                                                                   (interrupt(): pauses
                                                                                    until a human resumes
                                                                                    with a decision)

Key point: add_edge()/add_conditional_edges() into the SAME node are
independent OR-triggers, not a join -- an AND-join needs all sources listed
together in one add_edge([...]) call. reflection's join is exactly the 4
assessment agents it reads; figure_table's one-shot output and reflection's
"proceed" decision are combined via ready_for_synthesis, since a conditional
edge's dynamic target can't itself be a list-join entry (see nodes.py).
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from core.graph.nodes import ReviewGraphNodes
from core.graph.state import ReviewGraphState
from core.llm.llm_provider import get_llm
from core.llm.prompt_manager import PromptManager


def build_review_graph(llm=None, prompt_manager=None, checkpointer=None):
    """Builds and compiles the review graph.

    Args:
        llm: injected for testability; defaults to get_llm() (json_mode=True).
        prompt_manager: injected for testability; defaults to a real PromptManager().
        checkpointer: injected for testability/production swap (e.g. a real
            SqliteSaver for crash-safe resume); defaults to an in-memory saver,
            which is enough for Phase 1 (one server process, no restart-resume
            guarantee yet -- that's a Phase 2 concern alongside human-in-the-loop).

    Returns:
        A compiled LangGraph graph. Invoke with
        `graph.invoke({"parsed_paper": parsed_paper}, config={"configurable": {"thread_id": run_id}})`.
    """
    nodes = ReviewGraphNodes(
        llm=llm or get_llm(),
        prompt_manager=prompt_manager or PromptManager(),
    )

    graph = StateGraph(ReviewGraphState)

    graph.add_node("paper_understanding", nodes.paper_understanding)
    graph.add_node("literature_rag", nodes.literature_rag)
    graph.add_node("figure_table", nodes.figure_table)
    graph.add_node("novelty", nodes.novelty)
    graph.add_node("methodology", nodes.methodology)
    graph.add_node("citation", nodes.citation)
    graph.add_node("evidence_reproducibility", nodes.evidence_reproducibility)
    graph.add_node("reflection", nodes.reflection)
    graph.add_node("prepare_revision", nodes.prepare_revision)
    graph.add_node("ready_for_synthesis", nodes.ready_for_synthesis)
    graph.add_node("final_review", nodes.final_review)
    graph.add_node("human_approval", nodes.human_approval)

    # Stage 1: everything that only needs parsed_paper starts immediately.
    graph.add_edge(START, "paper_understanding")
    graph.add_edge(START, "literature_rag")
    graph.add_edge(START, "figure_table")
    graph.add_edge(START, "methodology")
    graph.add_edge(START, "evidence_reproducibility")

    # Stage 2: novelty needs BOTH paper_understanding and literature_context --
    # add_edge with a LIST of sources is what actually creates an AND-join in
    # LangGraph ("wait for ALL of these"); calling add_edge separately per
    # source only registers independent triggers, so the target can fire as
    # soon as ANY ONE of them completes -- confirmed the hard way (a test run
    # hit reflection before novelty_assessment existed in state, because
    # methodology/evidence_reproducibility finish one superstep earlier and
    # each had their own separate edge into it).
    graph.add_edge(["paper_understanding", "literature_rag"], "novelty")
    graph.add_edge("literature_rag", "citation")

    # Reflection waits for exactly the 4 assessments it actually reads -- one
    # list-form edge, not four separate add_edge calls (see note above).
    graph.add_edge(
        ["novelty", "methodology", "citation", "evidence_reproducibility"], "reflection"
    )

    # Bounded revision loop (see route_after_reflection / settings.reflection.max_revision_passes).
    graph.add_conditional_edges(
        "reflection", nodes.route_after_reflection,
        {"revise": "prepare_revision", "proceed": "ready_for_synthesis"},
    )
    graph.add_edge("prepare_revision", "novelty")
    graph.add_edge("prepare_revision", "methodology")
    graph.add_edge("prepare_revision", "citation")
    graph.add_edge("prepare_revision", "evidence_reproducibility")

    # figure_table never gets revised and has no other consumer. Its one-shot
    # output and the "proceed" decision are combined into a single real
    # AND-join here (see ready_for_synthesis's docstring for why this
    # indirection is necessary rather than two separate edges into final_review).
    graph.add_edge(["figure_table", "ready_for_synthesis"], "final_review")

    # Mandatory human-in-the-loop gate: the drafted recommendation is not
    # "issued" until a human resumes the interrupted run. interrupt() (in
    # nodes.human_approval) requires a checkpointer + a thread_id in the invoke
    # config -- both already present -- so the parked run can be resumed later.
    graph.add_edge("final_review", "human_approval")
    graph.add_edge("human_approval", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
