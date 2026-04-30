from flask import Flask, render_template, request, session

from neural_network.nn_classifier import classify, get_missing_followup
from neural_network.slot_extractor import extract_slots
from decision_tree.decision_tree import recommend

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

conversation_state = {
    "pending_intent": None,
    "slots": {},
}

def build_decision_inputs(intent, slots):
    if intent == "course_recommendation":
        return [
            "course_recommendation",
            slots.get("major"),
            slots.get("concentration") or "None",
            slots.get("load_preference"),
            slots.get("course_type"),
            slots.get("year"),
        ]
    if intent == "study_habits":
        return [
            "gpa_help",
            slots.get("gpa"),
            slots.get("major"),
            slots.get("concentration") or "None",
            slots.get("year"),
        ]
    return None

def run_pipeline(user_message):
    if conversation_state["pending_intent"] is None:
        intent_result = classify(user_message)
        intent = intent_result["intent"]

        if intent == "unknown":
            return "Sorry, I can only help with course recommendations and GPA advice."

        slots = extract_slots(user_message, intent)
        conversation_state["pending_intent"] = intent
        conversation_state["slots"] = slots
    else:
        intent = conversation_state["pending_intent"]
        new_slots = extract_slots(user_message, intent)
        for key, value in new_slots.items():
            if conversation_state["slots"].get(key) is None and value is not None:
                conversation_state["slots"][key] = value

    followup = get_missing_followup(
        conversation_state["slots"], conversation_state["pending_intent"]
    )
    if followup:
        return followup

    final_intent = conversation_state["pending_intent"]
    final_slots  = conversation_state["slots"]

    conversation_state["pending_intent"] = None
    conversation_state["slots"] = {}

    decision_inputs = build_decision_inputs(final_intent, final_slots)
    if decision_inputs is None:
        return "I could not process that request."

    return recommend(decision_inputs)

@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_message = request.form.get("message", "").strip()
        if user_message:
            bot_response = run_pipeline(user_message)
            session["chat_history"].append({
                "user": user_message,
                "bot":  bot_response
            })
            session.modified = True
        
    return render_template("index.html", chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    app.run(debug=True)