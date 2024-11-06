import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import vae, classifier

# Load a pre-trained classifier (e.g., simple CNN for MNIST or another dataset)
# Assuming classifier and data loader for adversarial examples (adv_loader) are defined

def evaluate_purified(classifier, vae, adv_loader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in adv_loader:
            # Purify the adversarial examples
            purified_data = vae.purify(vae, data)
            # Predict with the classifier
            output = classifier(purified_data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    print(f"Accuracy after purification: {100 * correct / total:.2f}%")

def fgsm_attack(model, data, target, epsilon):
    """Generates adversarial example using FGSM"""
    data.requires_grad = True
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    # Generate adversarial example by perturbing in the direction of the gradient
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    # Clip the values to ensure they stay within the valid range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# Transformation to normalize the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load MNIST training data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Load the MNIST test data
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Hyperparameters for adversarial example generation
epsilon = 0.1  # Controls the magnitude of the perturbation

# Assuming 'classifier' is your trained model and 'test_loader' is your clean test data loader
adv_examples = []
adv_targets = []

classifier.eval()  # Set classifier to evaluation mode

for data, target in test_loader:
    # Generate adversarial examples
    adv_data = fgsm_attack(classifier, data, target, epsilon)
    adv_examples.append(adv_data)
    adv_targets.append(target)

# Concatenate the adversarial examples and targets into single tensors
adv_examples = torch.cat(adv_examples, dim=0)
adv_targets = torch.cat(adv_targets, dim=0)

# Create a DataLoader from the adversarial examples
adv_dataset = torch.utils.data.TensorDataset(adv_examples, adv_targets)
adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=64, shuffle=False)

# Initialize VAE and classifier
vae = vae.VAE(input_dim=784, hidden_dim=400, z_dim=20)
# Assuming `classifier` and `adv_loader` are pre-defined

# Load and preprocess dataset
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                                         transform=transforms.ToTensor()),
                          batch_size=64, shuffle=True)

# Train VAE
vae.train_vae(vae, train_loader, epochs=10)

classifier = classifier.train_clf()
# Evaluate on clean test data
classifier.evaluate_clf(classifier, test_loader)

# Evaluate on adversarial examples
evaluate_purified(classifier, vae, adv_loader)

