import torch
import torchvision.models as models
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# ✅ Step 1: Load Pretrained Models
def load_pretrained_models(num_models=30):
    """Loads multiple pretrained MobileNetV3-Small models"""
    return [models.mobilenet_v3_small(pretrained=True) for _ in range(num_models)] # pretrained=True

models_list = load_pretrained_models()

# ✅ Step 2: Extract Model Metadata (Sparsity, Stability, Health)
def get_model_metadata(model):
    """Extracts sparsity, stability, and health statistics from a model's weights"""
    total_params = 0
    zero_params = 0
    weight_stability = []
    health_scores = []

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param.abs() < 1e-3).sum().item()  # Sparsity (near-zero weights)

            weight_stability.append(param.std().item())  # Stability = standard deviation
            health_scores.append((param.mean().item()) / (param.std().item() + 1e-5))  # Health metric

    sparsity = zero_params / total_params  # Percentage of near-zero weights
    stability = np.mean(weight_stability)
    health = np.mean(health_scores)

    return {"sparsity": sparsity, "stability": stability, "health": health}

# ✅ Step 3: Compute Metadata for Each Model
metadata_list = [get_model_metadata(model) for model in models_list]

# ✅ Step 4: Define Genetic Algorithm (GA)
def fitness_function(metadata):
    """Defines the fitness function using weighted sum of metadata."""
    return 0.4 * (1 - metadata["sparsity"]) + 0.4 * metadata["stability"] + 0.2 * metadata["health"]

def select_best_models(metadata_list, models_list, num_selected=2):
    """Selects best models using Genetic Algorithm based on fitness scores."""
    scores = [fitness_function(meta) for meta in metadata_list]
    sorted_indices = np.argsort(scores)[::-1]  # Sort by fitness (descending)
    return [models_list[i] for i in sorted_indices[:num_selected]]

# ✅ Step 5: Apply GA Selection
selected_models = select_best_models(metadata_list, models_list)

# ✅ Step 6: Aggregate Selected Models with FedAvg
def federated_averaging(models_list):
    """Aggregates multiple MobileNetV3 models using Federated Averaging (FedAvg)."""
    global_model = models.mobilenet_v3_small(pretrained=False)  # Create empty global model
    global_state_dict = global_model.state_dict()

    avg_state_dict = {key: torch.zeros_like(param, dtype=torch.float32) for key, param in global_state_dict.items()}
    num_models = len(models_list)

    for model in models_list:
        for key, param in model.state_dict().items():
            avg_state_dict[key] += param.float() / num_models  # Average weights

    global_model.load_state_dict(avg_state_dict)
    return global_model

fedavg_model = federated_averaging(models_list)
ga_model = federated_averaging(selected_models)  # FedAvg on GA-selected models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Step 7: Fine-tune GA Model
def fine_tune_model(model, train_loader, epochs=5):
    """Fine-tunes the GA-selected model."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy * 100:.2f}%")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset for training
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Fine-tune the GA-selected model
fine_tune_model(ga_model, train_loader)

# ✅ Step 8: Evaluate Models on CIFAR-10
# Load dataset

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################
global_model.to(device)

# ✅ Step 3: Freeze Lower Layers (Keep Pretrained Features)
for param in global_model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor

# ✅ Step 4: Modify Classifier for CIFAR-10 (10 Classes)
num_ftrs = global_model.classifier[0].in_features
global_model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10)  # CIFAR-10 has 10 classes
)
global_model.to(device)

##########################################################

criterion = nn.CrossEntropyLoss()

def evaluate_model(model, test_loader):
    """Computes loss and accuracy of a given model."""
    model.to(device)
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

            loss_list.append(loss.item())
            accuracy_list.append(correct / labels.size(0))

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples

    test_losses.append(avg_loss)
    test_accuracies.append(avg_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}: Test Loss={avg_loss:.4f}, Accuracy={avg_accuracy*100:.2f}%")

    return avg_loss, avg_accuracy, loss_list, accuracy_list

# Evaluate both models
fedavg_loss, fedavg_acc, fedavg_loss_list, fedavg_acc_list = evaluate_model(fedavg_model, test_loader)
ga_loss, ga_acc, ga_loss_list, ga_acc_list = evaluate_model(ga_model, test_loader)

# ✅ Step 9: Compare Performance
plt.figure(figsize=(12, 5))

# Loss Comparison
plt.subplot(1, 2, 1)
plt.plot(fedavg_loss_list, label="FedAvg Loss", color="red")
plt.plot(ga_loss_list, label="GA Loss", color="blue")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Loss Comparison: FedAvg vs. GA")
plt.legend()

# Accuracy Comparison
plt.subplot(1, 2, 2)
plt.plot([acc * 100 for acc in fedavg_acc_list], label="FedAvg Accuracy", color="red")
plt.plot([acc * 100 for acc in ga_acc_list], label="GA Accuracy", color="blue")
plt.xlabel("Batch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: FedAvg vs. GA")
plt.legend()

plt.show()

# ✅ Print Final Comparison
print(f"📉 FedAvg Final Loss: {fedavg_loss:.4f}, ✅ Accuracy: {fedavg_acc * 100:.2f}%")
print(f"📉 GA Final Loss: {ga_loss:.4f}, ✅ Accuracy: {ga_acc * 100:.2f}%")

# ✅ Conclusion
if ga_acc > fedavg_acc:
    print("🚀 Genetic Algorithm selected a better model than Federated Averaging!")
else:
    print("❗ Federated Averaging performed better than Genetic Algorithm selection.")
