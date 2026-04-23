import os, json, re
from anthropic import Anthropic
from langchain_core.messages import AIMessage, HumanMessage
from agent.state import (AgentState, LeadInfo, INTENT_GREETING, INTENT_PRODUCT_INQUIRY,
    INTENT_HIGH_INTENT, INTENT_PROVIDE_INFO, INTENT_OTHER, ALL_LEAD_FIELDS)
from agent.rag import retrieve_context, get_full_kb
from agent.tools import mock_lead_capture, validate_email, extract_lead_fields_from_text

_client = Anthropic()
_MODEL = "claude-haiku-4-5"

def _llm(system, messages, max_tokens=512):
    fmt = []
    for m in messages:
        if isinstance(m, HumanMessage):
            fmt.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            fmt.append({"role": "assistant", "content": m.content})
        elif isinstance(m, dict):
            fmt.append(m)
    if not fmt:
        fmt = [{"role": "user", "content": "Hello"}]
    r = _client.messages.create(model=_MODEL, max_tokens=max_tokens, system=system, messages=fmt)
    return r.content[0].text.strip()

def classify_intent(state: AgentState) -> dict:
    msgs = state["messages"]
    last = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
    lead_info = state.get("lead_info", {})
    pending = state.get("pending_fields", ALL_LEAD_FIELDS[:])
    prompt = f"""Classify the user message into ONE intent:
- greeting: hello, hi, hey
- product_inquiry: questions about features, pricing, plans, policies
- high_intent: ready to sign up, buy, try, subscribe
- provide_info: supplying name, email, or platform
- other: anything else

Pending lead fields: {pending}
Collected: {json.dumps(lead_info)}
Message: "{last}"

Reply ONLY with JSON: {{"intent": "<label>", "confidence": "<high|medium|low>"}}"""
    raw = _llm("Output only valid JSON.", [{"role": "user", "content": prompt}], 64)
    try:
        intent = json.loads(raw).get("intent", INTENT_OTHER)
    except:
        lo = last.lower()
        if any(w in lo for w in ["hi","hello","hey"]): intent = INTENT_GREETING
        elif any(w in lo for w in ["price","plan","feature","cost","refund"]): intent = INTENT_PRODUCT_INQUIRY
        elif any(w in lo for w in ["sign up","buy","try","subscribe","want to"]): intent = INTENT_HIGH_INTENT
        elif "@" in last or any(p in lo for p in ["youtube","instagram","tiktok"]): intent = INTENT_PROVIDE_INFO
        else: intent = INTENT_OTHER
    if pending and intent not in (INTENT_GREETING, INTENT_PRODUCT_INQUIRY):
        if extract_lead_fields_from_text(last, lead_info) != lead_info:
            intent = INTENT_PROVIDE_INFO
    return {"intent": intent}

def retrieve_rag(state: AgentState) -> dict:
    msgs = state["messages"]
    last = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
    if state.get("intent") == INTENT_GREETING:
        return {"rag_context": None}
    return {"rag_context": retrieve_context(last)}

_SYS = """You are Aria, the friendly AI sales assistant for AutoStream — an AI-powered video editing SaaS.
Be warm, professional, and concise. Only answer from the knowledge base provided.
Never ask for more than one lead field per turn. Order: name first, then email, then platform.
Knowledge Base:\n{kb}"""

def generate_response(state: AgentState) -> dict:
    msgs = state["messages"]
    intent = state.get("intent", INTENT_OTHER)
    kb = state.get("rag_context") or get_full_kb()
    lead_info = state.get("lead_info", {})
    pending = state.get("pending_fields", ALL_LEAD_FIELDS[:])
    captured = state.get("lead_captured", False)
    sys = _SYS.format(kb=kb)
    field_labels = {"name": "full name", "email": "email address", "platform": "content platform (YouTube, Instagram, TikTok, etc.)"}
    if intent == INTENT_HIGH_INTENT and not captured and pending:
        sys += f"\n\nUser has high purchase intent. Ask for their {field_labels[pending[0]]} now."
    if intent == INTENT_PROVIDE_INFO and pending:
        if len(pending) > 1:
            sys += f"\n\nAcknowledge what they shared, then ask for their {field_labels[pending[0]]}."
        else:
            sys += "\n\nThey have provided all details. Confirm and say the team will be in touch!"
    if captured:
        sys += "\n\nLead is captured. Wrap up warmly."
    history = []
    for m in msgs:
        if isinstance(m, HumanMessage): history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage): history.append({"role": "assistant", "content": m.content})
    resp = _llm(sys, history, 400)
    return {"messages": [AIMessage(content=resp)], "turn_count": state.get("turn_count", 0) + 1}

def collect_lead_info(state: AgentState) -> dict:
    msgs = state["messages"]
    lead_info = dict(state.get("lead_info", {}))
    last = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
    lead_info = extract_lead_fields_from_text(last, lead_info)
    if not lead_info.get("name") and state.get("intent") == INTENT_PROVIDE_INFO:
        raw = _llm("Extract the person full name or reply NONE.", [{"role":"user","content":f"Message: \"{last}\""}], 32)
        if raw.strip().upper() != "NONE" and len(raw.strip()) > 1:
            lead_info["name"] = raw.strip()
    pending = [f for f in ALL_LEAD_FIELDS if not lead_info.get(f)]
    return {"lead_info": LeadInfo(**lead_info), "pending_fields": pending}

def capture_lead(state: AgentState) -> dict:
    if state.get("lead_captured"): return {}
    li = state.get("lead_info", {})
    name, email, platform = li.get("name"), li.get("email"), li.get("platform")
    if not all([name, email, platform]): return {}
    if not validate_email(email): return {}
    mock_lead_capture(name=name, email=email, platform=platform)
    return {"lead_captured": True, "pending_fields": []}
