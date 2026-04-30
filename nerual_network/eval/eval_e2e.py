from nerual_network.nn_classifier import chat, session

def reset_session():
    session["pending_intent"] = None
    session["slots"] = {}

conversations = [
    {
        "turns": ["recommend me courses", "junior", "CSE", "hard", "AI", "elective"],
        "expect_keyword": "CSE"  # final response should mention a CSE course
    },
    {
        "turns": ["my GPA is low", "2.1", "sophomore", "CS", "None"],
        "expect_keyword": "GPA"
    }
]

for i, convo in enumerate(conversations):
    reset_session()
    response = ""
    for turn in convo["turns"]:
        response = chat(turn)
    keyword = convo["expect_keyword"]
    status = "PASS" if keyword in response else "FAIL"
    print(f"Conversation {i+1}: {status} (looked for '{keyword}' in final response)")
    print(f"  Final response: {response[:100]}...\n")