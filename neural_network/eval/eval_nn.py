import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from ..preprocess import build_vocab
from ..model import ChatbotNN

all_words, tags, X, Y = build_vocab()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

data = torch.load("chatbot_model.pth")
model = ChatbotNN(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

preds = []
for x in X_test:
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(out, dim=1).item()
    preds.append(pred)

# Classification report
print(classification_report(Y_test, preds, target_names=tags))

# Confusion matrix
cm = confusion_matrix(Y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=tags, yticklabels=tags)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved to confusion_matrix.png")