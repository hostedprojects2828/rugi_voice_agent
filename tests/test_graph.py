from app.graph import wake_up_graph


def run(message):
    return wake_up_graph.invoke({"user_message": message, "action": "", "agent_response": ""})


def test_tired():
    assert run("I'm tired, five more minutes")["action"] == "motivate"


def test_awake():
    assert run("I'm awake")["action"] == "finish"


def test_other():
    assert run("Good morning")["action"] == "conversation"
