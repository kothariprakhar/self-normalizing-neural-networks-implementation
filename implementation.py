import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import math

# ==========================================
# 1. Configuration & Setup
# ==========================================

# Configuration
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-4
DEPTH = 20  # Number of hidden layers (Deep FNN)
HIDDEN_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)

# ==========================================
# 2. Data Preparation (FashionMNIST)
# ==========================================

# We use FashionMNIST as a proxy for complex tabular/signal data.
# Flattening images treats them as high-dim vectors suitable for FNNs.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize inputs to [-1, 1]
])

# Download and Load
print(f"Downloading Data... (Device: {DEVICE})")
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Subset for speed if needed, though FashionMNIST is fast
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Model Architectures
# ==========================================

class SNN(nn.Module):
    """
    Self-Normalizing Neural Network
    Components:
    1. SELU Activation: Scaled Exponential Linear Unit
    2. LeCun Normal Initialization: Essential for SNN convergence
    3. AlphaDropout: Maintains mean/var during dropout
    """
    def __init__(self, input_dim, output_dim, hidden_dim, depth, dropout_rate=0.05):
        super(SNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        self.act = nn.SELU()
        self.dropout = nn.AlphaDropout(p=dropout_rate)
        
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Paper Requirement: Weights must be initialized with mean 0 and 
        std = sqrt(1 / input_size). This is 'LeCun Normal'.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # fan_in calculates the number of input units
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(1.0 / m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
            x = self.dropout(x)
        return self.output(x)

class StandardReLU(nn.Module):
    """
    Baseline Deep FNN with ReLU and He Initialization.
    Without Batch Norm, very deep ReLU networks often suffer from 
    dying neurons or exploding gradients.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, depth):
        super(StandardReLU, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        
        # He Initialization (Standard for ReLU)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        return self.output(x)

# ==========================================
# 4. Instrumentation (Hooks)
# ==========================================

# We use hooks to record the Mean and Std of activations at deep layers
# to prove the "Self-Normalizing" property.
stats_history = {"SNN": {"mean": [], "std": []}, "ReLU": {"mean": [], "std": []}}

def get_activation_hook(model_name, layer_idx):
    def hook(model, input, output):
        # Only record stats during validation to keep plots clean
        if not model.training:
            flat = output.detach().view(-1)
            stats_history[model_name]["mean"].append(flat.mean().item())
            stats_history[model_name]["std"].append(flat.std().item())
    return hook

# ==========================================
# 5. Training Loop
# ==========================================

def train_evaluate(model, model_name):
    print(f"\nTraining {model_name} (Depth {DEPTH})...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Register hook on the middle-deep layer (e.g., layer 15 of 20)
    # to see if signals vanish or explode deep in the network.
    probe_layer_idx = DEPTH // 2
    model.layers[probe_layer_idx].register_forward_hook(get_activation_hook(model_name, probe_layer_idx))

    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation pass (triggers hooks)
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {acc:.2f}%")
        
    return loss_history

# ==========================================
# 6. Execution & Visualization
# ==========================================

input_dim = 28 * 28
output_dim = 10

# Initialize Models
snn_model = SNN(input_dim, output_dim, HIDDEN_DIM, DEPTH).to(DEVICE)
relu_model = StandardReLU(input_dim, output_dim, HIDDEN_DIM, DEPTH).to(DEVICE)

# Train
snn_losses = train_evaluate(snn_model, "SNN")
relu_losses = train_evaluate(relu_model, "ReLU")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Loss Curves
axes[0].plot(snn_losses, label="SNN (SELU + AlphaDropout)", color='blue', linewidth=2)
axes[0].plot(relu_losses, label="Standard FNN (ReLU)", color='red', linestyle='--', linewidth=2)
axes[0].set_title(f"Training Loss (Depth {DEPTH})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Activation Mean (Deep Layer)
# We average the recorded batch stats for clarity
snn_means = [np.mean(stats_history["SNN"]["mean"][i:i+10]) for i in range(0, len(stats_history["SNN"]["mean"]), 10)]
relu_means = [np.mean(stats_history["ReLU"]["mean"][i:i+10]) for i in range(0, len(stats_history["ReLU"]["mean"]), 10)]

axes[1].plot(snn_means, color='blue', label="SNN")
axes[1].plot(relu_means, color='red', linestyle='--', label="ReLU")
axes[1].set_title("Mean of Activations (Deep Layer)")
axes[1].set_xlabel("Validation Steps")
axes[1].axhline(0, color='black', linewidth=1, linestyle=':')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Activation Variance (Deep Layer)
snn_stds = [np.mean(stats_history["SNN"]["std"][i:i+10]) for i in range(0, len(stats_history["SNN"]["std"]), 10)]
relu_stds = [np.mean(stats_history["ReLU"]["std"][i:i+10]) for i in range(0, len(stats_history["ReLU"]["std"]), 10)]

axes[2].plot(snn_stds, color='blue', label="SNN")
axes[2].plot(relu_stds, color='red', linestyle='--', label="ReLU")
axes[2].set_title("Std Dev of Activations (Deep Layer)")
axes[2].set_xlabel("Validation Steps")
axes[2].axhline(1, color='black', linewidth=1, linestyle=':')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("snn_analysis.png")
print("\nAnalysis plot saved to 'snn_analysis.png'.")
print("Notice how SNN activations stay closer to Mean=0, Std=1, enabling deep training.")