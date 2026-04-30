import torch
import numpy as np
import matplotlib.pyplot as plt
from ..preprocess import build_vocab
from ..model import ChatbotNN

all_words, tags, X, Y = build_vocab()

data = torch.load("chatbot_model.pth")
model = ChatbotNN(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

confidences = []
correct = []

for x, y in zip(X, Y):
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze()
        conf, pred = torch.max(probs, dim=0)
    confidences.append(conf.item())
    correct.append(pred.item() == y)

confidences = np.array(confidences)
correct = np.array(correct)

print(f"Average confidence:          {confidences.mean():.4f}")
print(f"Average confidence (correct): {confidences[correct].mean():.4f}")
print(f"Average confidence (wrong):   {confidences[~correct].mean():.4f}")
print(f"Below 0.6 threshold:          {(confidences < 0.6).sum()} samples")

plt.hist(confidences, bins=20, edgecolor='black')
plt.axvline(x=0.6, color='red', linestyle='--', label='Threshold (0.6)')
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Model Confidence Distribution")
plt.legend()
plt.savefig("confidence_dist.png")
print("Saved to confidence_dist.png")