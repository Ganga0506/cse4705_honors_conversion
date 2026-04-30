import torch
import numpy as np
from model import ChatbotNN
from preprocess import tokenize, bag_of_words
from slot_extractor import extract_slots
from decision_tree import recommend

# ---- Load saved model ----
data = torch.load("chatbot_model.pth")

model = ChatbotNN(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags      = data["tags"]

CONFIDENCE_THRESHOLD = 0.6

def classify(sentence):
    tokens = tokenize(sentence)
    bow    = bag_of_words(tokens, all_words)
    x      = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs  = torch.softmax(output, dim=1).squeeze()

    confidence, predicted_idx = torch.max(probs, dim=0)
    confidence = confidence.item()
    intent     = tags[predicted_idx.item()]

    if confidence < CONFIDENCE_THRESHOLD:
        intent = "unknown"

    return {
        "intent":        intent,
        "confidence":    round(confidence, 4),
        "probabilities": {
            tag: round(probs[i].item(), 4)
            for i, tag in enumerate(tags)
        }
    }

def classify_and_extract(sentence):
    intent_result = classify(sentence)
    intent        = intent_result["intent"]
    confidence    = intent_result["confidence"]

    if intent == "unknown":
        return {
            "intent": "unknown",
            "confidence": confidence,
            "response": "I'm not sure I understand. Try asking about course recommendations or study tips!"
        }

    slots = extract_slots(sentence, intent)
    return {
        "intent":     intent,
        "confidence": confidence,
        **slots
    }

# ---- Session state ----
session = {
    "pending_intent": None,
    "slots": {}
}

def get_missing_followup(slots, intent):
    if intent == "course_recommendation":
        if slots.get("year") is None:
            return "What year are you? (freshman, sophomore, junior, or senior)"
        if slots.get("major") is None:
            return "What's your major? (CS, CSE, or DSE)"
        if slots.get("load_preference") is None:
            return "Do you want a light, medium, or hard course load?"
        if slots.get("concentration") is None:
            return "Do you have a concentration? (e.g. AI, Cybersecurity, Software, Systems, etc.)"
        if slots.get("course_type") is None:
            return "Do you prefer core or elective courses?"

    if intent == "study_habits":
        if slots.get("gpa") is None:
            return "What's your current GPA?"
        if slots.get("year") is None:
            return "What year are you? (freshman, sophomore, junior, or senior)"
        if slots.get("major") is None:
            return "What's your major? (CS, CSE, or DSE)"
        if slots.get("concentration") is None:
            return "Do you have a concentration? (e.g. AI, Cybersecurity, Software, Systems, etc.)"

    return None

def chat(user_message):
    if session["pending_intent"] is not None:
        intent = session["pending_intent"]
        existing_slots = session["slots"]

        new_slots = extract_slots(user_message, intent)

        for key, val in new_slots.items():
            if existing_slots.get(key) is None and val is not None:
                existing_slots[key] = val

        session["slots"] = existing_slots

    else:
        intent_result = classify(user_message)
        intent = intent_result["intent"]

        if intent == "unknown":
            return "I'm not sure I understand. Try asking about course recommendations or study tips!"

        slots = extract_slots(user_message, intent)
        session["pending_intent"] = intent
        session["slots"] = slots

    followup = get_missing_followup(session["slots"], session["pending_intent"])
    if followup:
        return followup

    intent = session["pending_intent"]
    slots  = session["slots"]

    session["pending_intent"] = None
    session["slots"] = {}

    if intent == "course_recommendation":
        inputs = [
            "course_selection",
        slots.get("major"),
        slots.get("concentration"),
        slots.get("load_preference"),
        slots.get("course_type"),   
        slots.get("year")         
        ]

    elif intent == "study_habits":
        inputs = [
            "gpa_help",
            slots.get("gpa"),
            slots.get("major"),
            slots.get("concentration"),
            slots.get("year")
        ]

    return recommend(inputs)

if __name__ == "__main__":
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        response = chat(msg)
        print(f"Bot: {response}\n")