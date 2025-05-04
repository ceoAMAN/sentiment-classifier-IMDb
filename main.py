import torch
import torch.optim as optim
from torch import nn
from utils import get_dataloaders
from modals import rnn_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
embed_dim = 100
hidden_dim = 128
output_dim = 2
num_epochs = 5
learning_rate = 0.001
num_layers = 1

train_loader, test_loader, vocab = get_dataloaders(batch_size=batch_size)

model = rnn_model(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_text, batch_labels in train_loader:
        batch_text, batch_labels = batch_text.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_text)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

def evaluate_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_text, batch_labels in data_loader:
            batch_text, batch_labels = batch_text.to(device), batch_labels.to(device)
            outputs = model(batch_text)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

    return correct_predictions / total_predictions

test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")