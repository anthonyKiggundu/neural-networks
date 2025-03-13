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
import psutil
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n DEVICE:: ", device)

# Load Pretrained Models
def load_pretrained_models(num_models=15):
    """Loads multiple pretrained MobileNetV3-Small models"""
    return [models.mobilenet_v3_small(pretrained=True).to(device) for _ in range(num_models)]

models_list = load_pretrained_models()

criterion = nn.CrossEntropyLoss()

# CIFAR-10 Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Extract Model Metadata (Sparsity, Stability, Health)
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

# Compute Metadata for Each Model
metadata_list = [get_model_metadata(model) for model in models_list]

# Define Genetic Algorithm (GA)
def fitness_function(metadata):
    """Defines the fitness function using weighted sum of metadata."""
    return 0.5 * (1 - metadata["sparsity"]) + 0.3 * metadata["stability"] + 0.2 * metadata["health"]

def crossover(parent1, parent2):
    """Performs crossover between two parent models"""
    child1 = models.mobilenet_v3_small(pretrained=False).to(device)
    child2 = models.mobilenet_v3_small(pretrained=False).to(device)
    
    for key in parent1.state_dict().keys():
        if random.random() > 0.5:
            child1.state_dict()[key].copy_(parent1.state_dict()[key])
            child2.state_dict()[key].copy_(parent2.state_dict()[key])
        else:
            child1.state_dict()[key].copy_(parent2.state_dict()[key])
            child2.state_dict()[key].copy_(parent1.state_dict()[key])
    
    return child1, child2

def mutate(model, mutation_rate=0.01):
    """Performs mutation on a model"""
    for param in model.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param) * 0.1

def select_best_models(metadata_list, models_list, num_selected=2):
    """Selects best models using Genetic Algorithm based on fitness scores."""
    scores = [fitness_function(meta) for meta in metadata_list]
    sorted_indices = np.argsort(scores)[::-1]  # Sort by fitness (descending)
    
    selected_models = [models_list[i] for i in sorted_indices[:num_selected]]
    
    # Perform crossover and mutation to generate new models
    new_models = []
    for i in range(0, len(selected_models), 2): #2
        if i+1 < len(selected_models):
            child1, child2 = crossover(selected_models[i], selected_models[i+1])
            mutate(child1)
            mutate(child2)
            new_models.extend([child1, child2])
    
    return new_models

# Apply GA Selection
selected_models = select_best_models(metadata_list, models_list)


def federated_sgd(models_list, train_loader, epochs=1):
    """Aggregates multiple MobileNetV3 models using Federated SGD (FedSGD)."""
    global_model = models.mobilenet_v3_small(pretrained=False).to(device)  # Create empty global model
    global_model.to(device)
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for model in models_list:
            model.to(device)
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    return global_model


# Aggregate Selected Models with FedAvg
def federated_averaging(models_list):
    """Aggregates multiple MobileNetV3 models using Federated Averaging (FedAvg)."""
    global_model = models.mobilenet_v3_small(pretrained=False).to(device)  # Create empty global model
    global_state_dict = global_model.state_dict()

    avg_state_dict = {key: torch.zeros_like(param, dtype=torch.float32) for key, param in global_state_dict.items()}
    num_models = len(models_list)

    for model in models_list:
        for key, param in model.state_dict().items():
            avg_state_dict[key] += param.float() / num_models  # Average weights

    global_model.load_state_dict(avg_state_dict)
    return global_model


ga_model = federated_averaging(selected_models)
fedavg_model = federated_averaging(models_list)
fedsgd_model = federated_sgd(models_list, train_loader)


# Fine-tune the selected models
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
        print(f"FineTuning - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy * 100:.2f}%")


# Fine-tune the GA-selected model
fine_tune_model(ga_model, train_loader)

# Fine-tune the FedAvg model
fine_tune_model(fedavg_model, train_loader)

# Fine-tune the FedAvg model
fine_tune_model(fedsgd_model, train_loader)

# Evaluate Models on CIFAR-10
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# criterion = nn.CrossEntropyLoss()

def evaluate_model(model, test_loader):
    """Computes loss and accuracy of a given model."""
    model.to(device)
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    loss_list, accuracy_list = [], []

    with torch.no_grad():
        for epoch, (images, labels) in enumerate(test_loader, 1):
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

            # Print the loss and accuracy for the current epoch
            print(f"Evaluation - Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {correct / labels.size(0) * 100:.2f}%")

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy, loss_list, accuracy_list


# Measure energy, computational power, and complexity
def measure_performance(func, *args, **kwargs):
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)
    start_mem = psutil.virtual_memory().used

    result = func(*args, **kwargs)

    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=None)
    end_mem = psutil.virtual_memory().used

    duration = end_time - start_time
    avg_cpu = (start_cpu + end_cpu) / 2
    mem_usage = (end_mem - start_mem) / (1024 ** 2)  # Convert to MB

    return result, duration, avg_cpu, mem_usage

# Create FedAvg and FedSGD Models
fedavg_model, fedavg_duration, fedavg_cpu, fedavg_mem = measure_performance(federated_averaging, models_list)
fedsdg_model, fedsdg_duration, fedsdg_cpu, fedsdg_mem = measure_performance(federated_sgd, models_list, train_loader)
fedavgen_model, fedavgen_duration, fedavgen_cpu, fedavgen_mem = measure_performance(federated_averaging, selected_models)

# Evaluate all models
fedavg_loss, fedavg_acc, fedavg_loss_list, fedavg_acc_list = evaluate_model(fedavg_model, test_loader)
fedsdg_loss, fedsdg_acc, fedsdg_loss_list, fedsdg_acc_list = evaluate_model(fedsdg_model, test_loader)
ga_loss, ga_acc, ga_loss_list, ga_acc_list = evaluate_model(ga_model, test_loader)

# Compare Performance
plt.figure(figsize=(12, 5))

# Loss Comparison
plt.subplot(1, 2, 1)
plt.plot(fedavg_loss_list, label="FedAvg Loss", color="red")
plt.plot(fedsdg_loss_list, label="FedSGD Loss", color="green")
plt.plot(ga_loss_list, label="FedAvgen Loss", color="blue")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Loss Comparison: FedAvg vs. FedSGD vs. FedAvgen")
plt.legend()

# Accuracy Comparison
plt.subplot(1, 2, 2)
plt.plot([acc * 100 for acc in fedavg_acc_list], label="FedAvg Accuracy", color="red")
plt.plot([acc * 100 for acc in fedsdg_acc_list], label="FedSGD Accuracy", color="green")
plt.plot([acc * 100 for acc in ga_acc_list], label="FedAvgen Accuracy", color="blue")
plt.xlabel("Batch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: FedAvg vs. FedSGD vs. FedAvgen")
plt.legend()

plt.savefig("metric_compared_run2.pdf", dpi=300, bbox_inches="tight")

# Energy Consumption Comparison
labels = ['FedAvg', 'FedSGD', 'FedAvgen']
durations = [fedavg_duration, fedsdg_duration, fedavgen_duration] #ga_duration]
cpu_usages = [fedavg_cpu, fedsdg_cpu, fedavgen_cpu]
mem_usages = [fedavg_mem, fedsdg_mem, fedavgen_mem]

plt.subplot(2, 2, 3)
x = np.arange(len(labels))
width = 0.3

fig, ax1 = plt.subplots()
ax1.bar(x - width, durations, width, label='Duration (s)', color='skyblue')
ax1.set_xlabel('Approach')
ax1.set_ylabel('Duration (s)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.bar(x, cpu_usages, width, label='CPU Usage (%)', color='lightgreen')
ax2.set_ylabel('CPU Usage (%)')
ax2.legend(loc='upper center')

ax3 = ax1.twinx()
ax3.bar(x + width, mem_usages, width, label='Memory Usage (MB)', color='salmon')
ax3.set_ylabel('Memory Usage (MB)')
ax3.legend(loc='upper right')

fig.tight_layout()
plt.title("Energy Consumption, Computational Power, and Complexity")

plt.savefig("resources_compared.pdf", dpi=300, bbox_inches="tight")

plt.show()

# Print Final Comparison
print(f"📉 FedAvg Final Loss: {fedavg_loss:.4f}, ✅ Accuracy: {fedavg_acc * 100:.2f}%")
print(f"📉 FedSGD Final Loss: {fedsdg_loss:.4f}, ✅ Accuracy: {fedsdg_acc * 100:.2f}%")
print(f"📉 GA Final Loss: {ga_loss:.4f}, ✅ Accuracy: {ga_acc * 100:.2f}%")

print(f"⚡ FedAvg Duration: {fedavg_duration:.2f}s, CPU Usage: {fedavg_cpu:.2f}%, Memory Usage: {fedavg_mem:.2f}MB")
print(f"⚡ FedSGD Duration: {fedsdg_duration:.2f}s, CPU Usage: {fedsdg_cpu:.2f}%, Memory Usage: {fedsdg_mem:.2f}MB")
print(f"⚡ FedAvgen Duration: {fedavgen_duration:.2f}s, CPU Usage: {fedavgen_cpu:.2f}%, Memory Usage: {fedavgen_mem:.2f}MB")

# Conclusion
if ga_acc > fedavg_acc:
    print("🚀 Genetic Algorithm selected a better model than Federated Averaging!")
elif fedsdg_acc > fedavg_acc:
    print("🚀 FedSGD performed better than FedAvg!")
else:
    print("❗ Federated Averaging performed better than Genetic Algorithm selection.")
