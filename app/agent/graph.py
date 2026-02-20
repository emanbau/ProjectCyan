from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Annotated
from app.agent.tools import RESEARCH_TOOLS
from app.agent.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from app.core.config import settings
import operator

# ── State Schema ──────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages: Annotated[list, operator.add]   # Full message history
    cycle_id: str                              # Current research cycle ID
    cycle_count: int                           # How many cycles completed
    features_ruled_in: list[str]              # Cumulative feature rulebook
    features_ruled_out: list[str]
    promoted_strategies: list[str]            # Strategies ready for paper trading
    current_hypothesis: str                   # Active hypothesis being tested
    last_backtest_result: dict | None         # Most recent backtest output
    should_continue: bool                     # Whether to run another cycle

# ── Model Setup ───────────────────────────────────────────────────
llm = ChatAnthropic(
    model=settings.claude_model,              # "claude-opus-4-6"
    api_key=settings.anthropic_api_key,
    max_tokens=8192,
).bind_tools(RESEARCH_TOOLS)

# ── Node Definitions ──────────────────────────────────────────────
def orchestrator_node(state: ResearchState) -> ResearchState:
    """
    Main Claude Opus reasoning node.
    Receives full state context, reasons, calls tools, returns updated messages.
    """
    # Build context message summarising current state
    context = f"""
RESEARCH CYCLE: {state['cycle_count'] + 1}
CYCLE ID: {state['cycle_id']}

FEATURES RULED IN SO FAR: {', '.join(state['features_ruled_in']) or 'None yet'}
FEATURES RULED OUT SO FAR: {', '.join(state['features_ruled_out']) or 'None yet'}
PROMOTED STRATEGIES: {', '.join(state['promoted_strategies']) or 'None yet'}
CURRENT HYPOTHESIS: {state['current_hypothesis'] or 'None — start fresh'}
LAST BACKTEST: {state['last_backtest_result'] or 'No backtest run yet'}

Begin the research cycle. Follow the process in your system prompt.
"""

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "messages": [response],
        "cycle_count": state["cycle_count"] + 1,
    }


def tool_node(state: ResearchState) -> ResearchState:
    """Executes any tool calls Claude made"""
    tool_executor = ToolNode(RESEARCH_TOOLS)
    result = tool_executor.invoke(state)
    return {**state, "messages": result["messages"]}


def update_state_from_messages(state: ResearchState) -> ResearchState:
    """
    Parses Claude's last message to extract structured state updates.
    Updates ruled in/out features, promoted strategies, next hypothesis.
    """
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else ""

    # Simple keyword extraction — in production use structured output
    # Claude is prompted to use specific markers for parsing
    new_state = dict(state)

    if "RULED IN:" in content:
        ruled_in_line = content.split("RULED IN:")[1].split("\n")[0]
        new_features = [f.strip() for f in ruled_in_line.split(",") if f.strip()]
        new_state["features_ruled_in"] = list(set(
            state["features_ruled_in"] + new_features
        ))

    if "RULED OUT:" in content:
        ruled_out_line = content.split("RULED OUT:")[1].split("\n")[0]
        ruled_out = [f.strip() for f in ruled_out_line.split(",") if f.strip()]
        new_state["features_ruled_out"] = list(set(
            state["features_ruled_out"] + ruled_out
        ))

    if "NEXT HYPOTHESIS:" in content:
        new_state["current_hypothesis"] = content.split(
            "NEXT HYPOTHESIS:"
        )[1].split("\n")[0].strip()

    if "PROMOTE" in content:
        # Extract strategy name and add to promoted list
        pass  # TODO: parse strategy name

    return new_state


def should_continue_research(state: ResearchState) -> str:
    """
    Routing function — decides whether to run another cycle or end.
    """
    last_message = state["messages"][-1]

    # If Claude made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # If max cycles reached, end
    if state["cycle_count"] >= 10:
        return "end"

    # If Claude flagged continuation
    if state.get("should_continue", True):
        return "continue"

    return "end"


# ── Graph Assembly ────────────────────────────────────────────────
def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", tool_node)
    graph.add_node("update_state", update_state_from_messages)

    # Entry point
    graph.set_entry_point("orchestrator")

    # Conditional routing from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        should_continue_research,
        {
            "tools": "tools",
            "continue": "update_state",
            "end": END,
        }
    )

    # After tools execute, return to orchestrator
    graph.add_edge("tools", "orchestrator")

    # After state update, loop back for next cycle
    graph.add_edge("update_state", "orchestrator")

    return graph.compile()


# ── Entry Point ───────────────────────────────────────────────────
def run_research_loop(initial_hypothesis: str = ""):
    """
    Start or resume the autonomous research loop.
    Can be called on a schedule (e.g., every hour via APScheduler).
    """
    import uuid
    graph = build_research_graph()

    initial_state: ResearchState = {
        "messages": [],
        "cycle_id": str(uuid.uuid4()),
        "cycle_count": 0,
        "features_ruled_in": [],
        "features_ruled_out": [],
        "promoted_strategies": [],
        "current_hypothesis": initial_hypothesis,
        "last_backtest_result": None,
        "should_continue": True,
    }

    for event in graph.stream(initial_state, stream_mode="values"):
        # Stream outputs in real time — useful for monitoring
        if "messages" in event and event["messages"]:
            last = event["messages"][-1]
            if hasattr(last, "content"):
                print(f"\n[Cycle {event.get('cycle_count', '?')}] {last.content[:500]}...")