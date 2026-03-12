import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET_DIR = r"C:\anveshak\cone_depression\data_main" 
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
IMG_SIZE = 128  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture ---
class RobustCNN(nn.Module):
    def __init__(self, num_classes):
        super(RobustCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)) 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- Training Logic ---
if __name__ == "__main__":
    # Augmentations for reliability
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Path {DATASET_DIR} not found.")
    else:
        full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
        num_classes = len(full_dataset.classes)

        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        torch.manual_seed(42) 
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = RobustCNN(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        best_acc = 0.0
        print(f"Training on {DEVICE}...")
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Save metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation Step
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * val_correct / val_total
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}: Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "model.pth")

        plt.figure(figsize=(12, 5))
        
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Training Acc')
        plt.plot(history['test_acc'], label='Test Acc')
        plt.title('Accuracy VS Epochs')
        plt.legend()

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.title('Loss VS Epochs')
        plt.legend()

        plt.savefig("training_curves.png")
        plt.show()