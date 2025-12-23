import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from chess_vision.model.model import SimpleSquareClassifier
from chess_vision.dataset.loaders import load_dataset

def main():
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleSquareClassifier(num_classes=2).to(device)
    print(f"Using: {device}")
    # Datasets
    train_loader, val_loader = load_dataset()
    # Epoch
    num_epochs = 10
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("\n=== Started Training ===\n")

    # Training Loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        torch.save(model.state_dict(), f"models/checkpoint_epoch{epoch+1}.pt")

        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Val Accuracy: {accuracy:.4f}")


if __name__=='__main__':
    main()