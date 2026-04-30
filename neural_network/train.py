import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neural_network.model import ChatbotNN
from neural_network.preprocess import build_vocab
import os

all_words, tags, X_train, Y_train = build_vocab()

# ---- Hyperparameters ----
INPUT_SIZE    = len(all_words)
HIDDEN_SIZE   = 128          # changed from 64
OUTPUT_SIZE   = len(tags)
EPOCHS        = 200          # changed from 300
BATCH_SIZE    = 8
LEARNING_RATE = 0.001
PATIENCE      = 20           # stop if no improvement for 20 epochs

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

# ---- Training Loop with early stopping ----
best_loss      = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    epoch_loss = 0
    for X_batch, Y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_no_improve = 0
        # Save best model so far
        torch.save({
            "model_state": model.state_dict(),
            "input_size":  INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
            "all_words":   all_words,
            "tags":        tags
        }, "chatbot_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} — no improvement for {PATIENCE} epochs")
            break

print(f"Training complete! Best loss: {best_loss:.4f}")
print("Model saved to chatbot_model.pth")