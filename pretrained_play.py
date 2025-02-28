import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

# ✅ Step 1: Load Pretrained MobileNetV3-Small Models
def load_pretrained_models(num_models=3):
    """Loads multiple pretrained MobileNetV3-Small models"""
    models_list = [models.mobilenet_v3_small(pretrained=True) for _ in range(num_models)]
    return models_list

models_list = load_pretrained_models()

# ✅ Step 2: Federated Averaging (FedAvg)
def federated_averaging(models_list):
    """Aggregates multiple MobileNetV3 models using Federated Averaging (FedAvg)."""
    global_model = models.mobilenet_v3_small(pretrained=False)  # Create empty global model
    global_state_dict = global_model.state_dict()

    # Initialize empty weight dictionary
    avg_state_dict = {key: torch.zeros_like(param) for key, param in global_state_dict.items()}

    num_models = len(models_list)
    for model in models_list:
        for key, param in model.state_dict().items():
            avg_state_dict[key] += param / num_models  # Average weights

    # Load averaged weights into global model
    global_model.load_state_dict(avg_state_dict)
    return global_model

# Aggregate models
global_model = federated_averaging(models_list)

# ✅ Step 3: Load Dataset (CIFAR-10 for testing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to MobileNetV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# ✅ Step 4: Evaluate Global Model (Loss & Accuracy)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model.to(device)
criterion = nn.CrossEntropyLoss()

def evaluate_model(model, test_loader):
    """Computes loss and accuracy of the global model on test data."""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    loss_list, accuracy_list = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

            # Store loss & accuracy for plotting
            loss_list.append(loss.item())
            accuracy_list.append(correct / labels.size(0))

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy, loss_list, accuracy_list

# Run evaluation
avg_loss, avg_accuracy, loss_list, accuracy_list = evaluate_model(global_model, test_loader)

# ✅ Step 5: Plot Loss & Accuracy Curves
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Test Loss", color="red")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("FedAvg Test Loss")
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(accuracy_list, label="Test Accuracy", color="blue")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.title("FedAvg Test Accuracy")
plt.legend()

plt.show()

# ✅ Print Final Performance
print(f"📉 Final Test Loss: {avg_loss:.4f}")
print(f"✅ Final Test Accuracy: {avg_accuracy * 100:.2f}%")
