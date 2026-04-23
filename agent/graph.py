from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState, ALL_LEAD_FIELDS
from agent.nodes import classify_intent, retrieve_rag, generate_response, collect_lead_info, capture_lead

def _should_capture(state: AgentState) -> str:
    if state.get("lead_captured"): return "respond"
    li = state.get("lead_info", {})
    pending = state.get("pending_fields", ALL_LEAD_FIELDS[:])
    if all(li.get(f) for f in ALL_LEAD_FIELDS) and not pending: return "capture"
    return "respond"

def build_graph():
    b = StateGraph(AgentState)
    b.add_node("classify_intent", classify_intent)
    b.add_node("retrieve_rag", retrieve_rag)
    b.add_node("collect_lead_info", collect_lead_info)
    b.add_node("capture_lead", capture_lead)
    b.add_node("generate_response", generate_response)
    b.add_edge(START, "classify_intent")
    b.add_edge("classify_intent", "retrieve_rag")
    b.add_edge("retrieve_rag", "collect_lead_info")
    b.add_conditional_edges("collect_lead_info", _should_capture, {"capture": "capture_lead", "respond": "generate_response"})
    b.add_edge("capture_lead", "generate_response")
    b.add_edge("generate_response", END)
    return b.compile(checkpointer=MemorySaver())
