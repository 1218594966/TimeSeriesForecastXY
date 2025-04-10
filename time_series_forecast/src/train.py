import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, model_name, train_loader, epochs=10, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
    return model, train_losses
