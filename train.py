import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocess import X_train, Y_train, all_words, tags
from model import ChatbotNN

# ---- Hyperparameters ----
INPUT_SIZE  = len(all_words)   
HIDDEN_SIZE = 64
OUTPUT_SIZE = len(tags)     
EPOCHS      = 300
BATCH_SIZE  = 8
LEARNING_RATE = 0.001

# ---- Dataset ----
class IntentDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor(X_train, dtype=torch.float32)
        self.y = torch.tensor(Y_train, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = IntentDataset()
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model, Loss, Optimizer ----
model     = ChatbotNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    for X_batch, Y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

print("Training complete!")

# ---- Save everything your classifier will need ----
torch.save({
    "model_state": model.state_dict(),
    "input_size":  INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "all_words":   all_words,
    "tags":        tags
}, "chatbot_model.pth")

print("Model saved to chatbot_model.pth")