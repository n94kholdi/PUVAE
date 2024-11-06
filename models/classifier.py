import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # Input channels=1 for grayscale (MNIST), 32 filters
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 64 filters for second conv layer
        self.fc1 = nn.Linear(12*12*64, 128)   # Adjust dimension based on conv output
        self.fc2 = nn.Linear(128, 10)         # Output layer for 10 classes (MNIST digits)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 12*12*64)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_clf(train_loader):

    # Initialize the model, loss function, and optimizer
    classifier = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

    return classifier

def evaluate_clf(classifier, data_loader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = classifier(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    print(f'Accuracy on clean test data: {100 * correct / total:.2f}%')


