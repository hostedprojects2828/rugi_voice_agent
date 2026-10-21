from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class WakeUpState(TypedDict):
    user_message: str
    action: str
    agent_response: str


def classify(state: WakeUpState) -> dict:
    message = state["user_message"].lower().replace("’", "'")
    if any(s in message for s in ("i'm awake", "im awake", "i am awake", "i'm up", "im up", "i am up")):
        action = "finish"
    elif any(s in message for s in ("five more minutes", "5 more minutes", "let me sleep", "i'm tired", "im tired", "i am tired")):
        action = "motivate"
    else:
        action = "conversation"
    return {"action": action}


def motivate(state: WakeUpState) -> dict:
    return {"agent_response": "Good morning! I know you're tired. Let's start small: sit up and put your feet on the floor. Can you do that for me?"}


def conversation(state: WakeUpState) -> dict:
    return {"agent_response": "Good morning! It's time to get moving. How are you feeling today?"}


def finish(state: WakeUpState) -> dict:
    return {"agent_response": "Brilliant, you're awake! Have some water, wash your face, and enjoy your morning!"}


builder = StateGraph(WakeUpState)
builder.add_node("classify", classify)
builder.add_node("motivate", motivate)
builder.add_node("conversation", conversation)
builder.add_node("finish", finish)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", lambda state: state["action"], {
    "motivate": "motivate", "conversation": "conversation", "finish": "finish"
})
for node in ("motivate", "conversation", "finish"):
    builder.add_edge(node, END)
wake_up_graph = builder.compile()
