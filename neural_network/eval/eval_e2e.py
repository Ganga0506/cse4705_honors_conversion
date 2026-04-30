import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from ..nn_classifier import chat, session

def reset_session():
    session["pending_intent"] = None
    session["slots"]          = {}

conversations = [
    {
        "turns":           ["recommend me courses", "junior", "CSE", "hard", "AI", "elective"],
        "expect_keyword":  "CSE",
        "label":           "course rec — junior CSE AI hard elective"
    },
    {
        "turns":           ["my GPA is low", "2.1", "sophomore", "CS", "no"],
        "expect_keyword":  "GPA",
        "label":           "study habits — 2.1 GPA sophomore CS no concentration"
    },
    {
        "turns":           ["what should i take", "freshman", "DSE", "light", "core"],
        "expect_keyword":  "CSE",
        "label":           "course rec — freshman DSE light core"
    },
    {
        "turns":           ["how do i improve my grades", "3.0", "senior", "CSE", "Cybersecurity"],
        "expect_keyword":  "GPA",
        "label":           "study habits — 3.0 GPA senior CSE cybersecurity"
    },
]

passed = 0
for i, convo in enumerate(conversations):
    reset_session()
    response = ""
    for turn in convo["turns"]:
        response = chat(turn)
    keyword = convo["expect_keyword"]
    status  = "PASS" if keyword in response else "FAIL"
    if status == "PASS":
        passed += 1
    print(f"Conversation {i+1} [{convo['label']}]: {status}")
    print(f"  Final response: {response[:120]}...\n")
print(f"{passed}/{len(conversations)} conversations passed")