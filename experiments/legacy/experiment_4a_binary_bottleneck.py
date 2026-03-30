"""
Experiment 4A: Binary Bottleneck on MNIST
========================================
Use binary features (via straight-through estimator) instead of linear projection.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_lambda(epoch):
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


class StraightThrough(torch.autograd.Function):
    """Straight-through estimator: round in forward, identity in backward."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryBottleneck(nn.Module):
    """Project to binary features using straight-through estimator."""
    def __init__(self, input_dim, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.project = nn.Linear(input_dim, n_bits)
    
    def forward(self, x):
        x = self.project(x)
        x = torch.sigmoid(x)  # Map to [0, 1]
        x = StraightThrough.apply(x)  # Binary: 0 or 1
        return x


class HolonomyChain(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.bit_phases = nn.Parameter(
            torch.ones(n_bits) * math.pi + torch.randn(n_bits) * 0.1
        )
        self.phi_0 = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.ones(n_bits) * math.pi)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        phi = self.phi_0.expand(batch_size)
        for i in range(self.n_bits):
            phi = phi + inputs[:, i] * self.bit_phases[i]
        output = (1.0 - torch.cos(phi)) / 2.0
        return output
    
    def compute_holonomy_loss(self):
        R_actual = torch.exp(1j * self.bit_phases)
        R_predicted = torch.exp(1j * self.A)
        return torch.mean(torch.abs(R_actual - R_predicted).pow(2))
    
    def get_phases(self):
        return self.bit_phases.detach().cpu().numpy()


class BinaryBottleneckClassifier(nn.Module):
    """Full classifier with binary bottleneck."""
    def __init__(self, input_dim=784, n_bits=16):
        super().__init__()
        self.n_bits = n_bits
        self.bottleneck = BinaryBottleneck(input_dim, n_bits)
        self.holonomy = HolonomyChain(n_bits)
    
    def forward(self, x):
        x = self.bottleneck(x)
        return self.holonomy(x)
    
    def compute_holonomy_loss(self):
        return self.holonomy.compute_holonomy_loss()
    
    def get_bit_phases(self):
        return self.holonomy.get_phases()


class LinearProjectClassifier(nn.Module):
    """Linear projection + HolonomyChain (original)."""
    def __init__(self, input_dim=784, n_bits=16):
        super().__init__()
        self.n_bits = n_bits
        self.project = nn.Linear(input_dim, n_bits)
        self.holonomy = HolonomyChain(n_bits)
    
    def forward(self, x):
        x = self.project(x)
        return self.holonomy(x)
    
    def compute_holonomy_loss(self):
        return self.holonomy.compute_holonomy_loss()
    
    def get_bit_phases(self):
        return self.holonomy.get_phases()


def load_mnist_oddeven(n_train=2000, n_test=500):
    """Load MNIST and create odd/even binary classification task."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_data = train_ds.data.float() / 255.0
    test_data = test_ds.data.float() / 255.0
    train_labels = (train_ds.targets % 2).float()
    test_labels = (test_ds.targets % 2).float()
    
    np.random.seed(42)
    train_indices = np.random.choice(len(train_ds), min(n_train, len(train_ds)), replace=False)
    test_indices = np.random.choice(len(test_ds), min(n_test, len(test_ds)), replace=False)
    
    class BinaryMNIST(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            x = self.data[idx].flatten().float() / 255.0
            y = self.labels[idx]
            return x, y
    
    train_subset = BinaryMNIST(train_data[train_indices], train_labels[train_indices])
    test_subset = BinaryMNIST(test_data[test_indices], test_labels[test_indices])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            bce = nn.functional.binary_cross_entropy(outputs, y)
            hol_loss = model.compute_holonomy_loss()
            lam = get_lambda(epoch) * 2.0
            loss = bce + lam * hol_loss
            loss.backward()
            optimizer.step()
        
        if epoch % 25 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in test_loader:
                    outputs = model(X)
                    preds = (outputs > 0.5)
                    correct += (preds == y).sum().item()
                    total += len(y)
            acc = correct / total
            print(f"  Epoch {epoch}: test_acc={acc:.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            preds = (outputs > 0.5)
            correct += (preds == y).sum().item()
            total += len(y)
    
    return correct / total


def main():
    print("="*60)
    print("EXPERIMENT 4A: BINARY BOTTLENECK ON MNIST")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST data...")
    train_loader, test_loader = load_mnist_oddeven(n_train=2000, n_test=500)
    
    input_dim = 784
    results = {}
    
    for n_bits in [16, 32]:
        print(f"\n{'='*50}")
        print(f"Testing Binary Bottleneck with n_bits={n_bits}")
        print(f"{'='*50}")
        
        # Binary bottleneck model
        model = BinaryBottleneckClassifier(input_dim=input_dim, n_bits=n_bits)
        print(f"Testing Binary Bottleneck...")
        test_acc_binary = train_and_evaluate(model, train_loader, test_loader, epochs=200, lr=0.01)
        phases_binary = model.get_bit_phases()
        mean_binary = np.mean(phases_binary)
        std_binary = np.std(phases_binary)
        
        print(f"\nBinary Bottleneck n={n_bits}:")
        print(f"  Test accuracy: {test_acc_binary:.4f}")
        print(f"  Phase mean:    {mean_binary:.4f}")
        print(f"  Phase std:     {std_binary:.4f}")
        
        results[f'binary_{n_bits}'] = {
            'test_acc': test_acc_binary,
            'phase_mean': mean_binary,
            'phase_std': std_binary,
            'phases': phases_binary.tolist()
        }
        
        # Linear projection model (for comparison)
        print(f"\nTesting Linear Projection (baseline)...")
        model_linear = LinearProjectClassifier(input_dim=input_dim, n_bits=n_bits)
        test_acc_linear = train_and_evaluate(model_linear, train_loader, test_loader, epochs=200, lr=0.01)
        phases_linear = model_linear.get_bit_phases()
        mean_linear = np.mean(phases_linear)
        std_linear = np.std(phases_linear)
        
        print(f"\nLinear Projection n={n_bits}:")
        print(f"  Test accuracy: {test_acc_linear:.4f}")
        print(f"  Phase mean:    {mean_linear:.4f}")
        print(f"  Phase std:     {std_linear:.4f}")
        
        results[f'linear_{n_bits}'] = {
            'test_acc': test_acc_linear,
            'phase_mean': mean_linear,
            'phase_std': std_linear,
            'phases': phases_linear.tolist()
        }
        
        # Plot phase distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(phases_binary, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=math.pi, color='r', linestyle='--', label=f'pi={math.pi:.2f}')
        axes[0].set_xlabel('Phase (radians)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Binary Bottleneck n={n_bits}')
        axes[0].legend()
        
        axes[1].hist(phases_linear, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=math.pi, color='r', linestyle='--', label=f'pi={math.pi:.2f}')
        axes[1].set_xlabel('Phase (radians)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Linear Projection n={n_bits}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'mnist_binary_vs_linear_n{n_bits}.png', dpi=150)
        plt.close()
        print(f"Saved plot to mnist_binary_vs_linear_n{n_bits}.png")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print("\n| Architecture        | n_bits | Test Acc | Mean Phase | Phase Std |")
    print("|---------------------|--------|----------|------------|-----------|")
    for n_bits in [16, 32]:
        b = results[f'binary_{n_bits}']
        l = results[f'linear_{n_bits}']
        print(f"| Binary Bottleneck   |   {n_bits:2d}   |  {b['test_acc']:.4f}  |   {b['phase_mean']:7.4f}  |  {b['phase_std']:.4f}  |")
        print(f"| Linear Projection   |   {n_bits:2d}   |  {l['test_acc']:4f}  |   {l['phase_mean']:7.4f}  |  {l['phase_std']:.4f}  |")
    
    # Save results
    with open('mnist_binary_bottleneck_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to mnist_binary_bottleneck_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
