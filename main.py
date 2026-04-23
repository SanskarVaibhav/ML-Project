import os, sys, uuid, argparse
from langchain_core.messages import HumanMessage, AIMessage

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Run: $env:ANTHROPIC_API_KEY=\"sk-ant-...\"")
    sys.exit(1)

from agent.graph import build_graph
from agent.state import ALL_LEAD_FIELDS

def run_chat(graph, thread_id, user_input):
    config = {"configurable": {"thread_id": thread_id}}
    existing = graph.get_state(config)
    if existing.values:
        inputs = {"messages": [HumanMessage(content=user_input)]}
    else:
        inputs = {"messages": [HumanMessage(content=user_input)], "lead_info": {},
                  "pending_fields": ALL_LEAD_FIELDS[:], "lead_captured": False,
                  "rag_context": None, "turn_count": 0}
    final = None
    for chunk in graph.stream(inputs, config=config, stream_mode="values"):
        final = chunk
    response = ""
    if final and "messages" in final:
        for m in reversed(final["messages"]):
            if isinstance(m, AIMessage):
                response = m.content
                break
    return response, final or {}

DEMO = [
    "Hi! I am looking for a video editing tool for my channel.",
    "Tell me about your pricing plans.",
    "What is included in the Pro plan? Do I get AI captions?",
    "What is your refund policy?",
    "That sounds great! I want to sign up for the Pro plan for my YouTube channel.",
    "My name is Alex Rivera",
    "alex.rivera@gmail.com",
    "YouTube",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--thread", type=str)
    args = parser.parse_args()
    thread_id = args.thread or str(uuid.uuid4())
    graph = build_graph()

    print("\n" + "="*55)
    print("   AutoStream - Inflx AI Sales Agent")
    print("   Type 'exit' to quit")
    print("="*55 + "\n")

    turns = DEMO if args.demo else []

    if args.demo:
        for msg in turns:
            print(f"You: {msg}")
            resp, state = run_chat(graph, thread_id, msg)
            print(f"Aria: {resp}\n")
            if state.get("lead_captured"):
                print("DEMO COMPLETE - Lead captured!")
                break
    else:
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user: continue
            if user.lower() in ("exit","quit"): break
            resp, state = run_chat(graph, thread_id, user)
            print(f"\nAria: {resp}\n")
            if state.get("lead_captured"):
                print("Your details have been sent to the AutoStream team!\n")

if __name__ == "__main__":
    main()
