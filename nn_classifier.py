import torch
import numpy as np
from model import ChatbotNN
from preprocess import tokenize, bag_of_words

# ---- Load saved model ----
data = torch.load("chatbot_model.pth")

model = ChatbotNN(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags      = data["tags"]

CONFIDENCE_THRESHOLD = 0.75

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

    # If not confident enough, return unknown
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

# ---- Test it ----
if __name__ == "__main__":
    test_inputs = [
        "what classes should I take?",
        "I am failing my courses",
        "recommend me some easy courses",
        "how do I study better?"
    ]
    for text in test_inputs:
        result = classify(text)
        print(f"\nInput: '{text}'")
        print(f"  Intent:      {result['intent']}")
        print(f"  Confidence:  {result['confidence']}")
        print(f"  Probs:       {result['probabilities']}")