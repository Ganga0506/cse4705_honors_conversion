from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nerual_network.preprocess import build_vocab
import torch
from nerual_network.model import ChatbotNN

all_words, tags, X_train, Y_train = build_vocab()
X_test, _, Y_test, _ = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

data = torch.load("chatbot_model.pth")
model = ChatbotNN(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

preds = []
for x in X_test:
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    preds.append(pred)

print(classification_report(Y_test, preds, target_names=tags))