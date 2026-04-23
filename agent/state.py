from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class LeadInfo(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]
    lead_info: LeadInfo
    lead_captured: bool
    pending_fields: List[str]
    rag_context: Optional[str]
    turn_count: int

INTENT_GREETING        = "greeting"
INTENT_PRODUCT_INQUIRY = "product_inquiry"
INTENT_HIGH_INTENT     = "high_intent"
INTENT_PROVIDE_INFO    = "provide_info"
INTENT_OTHER           = "other"
ALL_LEAD_FIELDS = ["name", "email", "platform"]
