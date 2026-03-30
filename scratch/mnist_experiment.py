"""
MNIST Binary Classification with HolonomyChain
==============================================
Task: Classify odd vs even digits (Z_2 structure)
Architecture: 784 -> Linear -> n_bits -> HolonomyChain -> output
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
from torch.utils.data import DataLoader, Subset

def get_lambda(epoch):
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


class HolonomyChain(nn.Module):
    """HolonomyChain for binary classification."""
    
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


class HolonomyClassifier(nn.Module):
    """Full classifier: Linear projection + HolonomyChain."""
    
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


class StandardMLP(nn.Module):
    """MLP baseline with same parameter count."""
    
    def __init__(self, input_dim=784, n_params=None):
        super().__init__()
        if n_params is None:
            hidden = 64
        else:
            hidden = max(1, (n_params - input_dim - 1) // (input_dim + 1))
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_mnist_oddeven(n_train=2000, n_test=500):
    """Load MNIST and create odd/even binary classification task."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Convert to float and normalize
    train_data = train_ds.data.float() / 255.0
    test_data = test_ds.data.float() / 255.0
    
    # Create binary labels: 0=even, 1=odd
    train_labels = (train_ds.targets % 2).float()
    test_labels = (test_ds.targets % 2).float()
    
    # Subsample for speed
    np.random.seed(42)
    train_indices = np.random.choice(len(train_ds), min(n_train, len(train_ds)), replace=False)
    test_indices = np.random.choice(len(test_ds), min(n_test, len(test_ds)), replace=False)
    
    # Create custom dataset
    class BinaryMNIST(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # Flatten: 28x28 -> 784
            x = self.data[idx].flatten().float() / 255.0
            y = self.labels[idx]
            return x, y
    
    train_subset = BinaryMNIST(train_data[train_indices], train_labels[train_indices])
    test_subset = BinaryMNIST(test_data[test_indices], test_labels[test_indices])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_holonomy(model, train_loader, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'accuracy': [], 'holonomy_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            bce = nn.functional.binary_cross_entropy(outputs, y.float())
            hol_loss = model.compute_holonomy_loss()
            lam = get_lambda(epoch) * 2.0
            loss = bce + lam * hol_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            acc = ((outputs > 0.5) == y).float().mean().item()
            total_acc += acc
            n_batches += 1
        
        history['loss'].append(total_loss / n_batches)
        history['accuracy'].append(total_acc / n_batches)
        history['holonomy_loss'].append(hol_loss.item())
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}, acc={total_acc/n_batches:.4f}")
    
    return history


def train_mlp(model, train_loader, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = nn.functional.binary_cross_entropy(outputs, y.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            acc = ((outputs > 0.5) == y).float().mean().item()
            total_acc += acc
            n_batches += 1
        
        history['loss'].append(total_loss / n_batches)
        history['accuracy'].append(total_acc / n_batches)
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}, acc={total_acc/n_batches:.4f}")
    
    return history


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            preds = (outputs > 0.5)
            correct += (preds == y).sum().item()
            total += len(y)
    
    return correct / total


def main():
    print("="*60)
    print("MNIST ODD/EVEN CLASSIFICATION EXPERIMENT")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST data...")
    train_loader, test_loader = load_mnist_oddeven(n_train=2000, n_test=500)
    
    # Get a sample to check dimensions
    input_dim = 784  # 28x28 flattened
    print(f"Input dimension: {input_dim}")
    
    # Results storage
    results = {}
    
    # Test different n_bits
    for n_bits in [16, 32]:
        print(f"\n{'='*50}")
        print(f"Testing HolonomyChain with n_bits={n_bits}")
        print(f"{'='*50}")
        
        # Create model
        model = HolonomyClassifier(input_dim=input_dim, n_bits=n_bits)
        n_params = count_params(model)
        print(f"Holonomy params: {n_params}")
        
        # Train
        print("\nTraining HolonomyChain...")
        history = train_holonomy(model, train_loader, epochs=200, lr=0.01)
        
        # Evaluate
        test_acc = evaluate(model, test_loader)
        train_acc = evaluate(model, train_loader)
        
        # Get learned phases
        phases = model.get_bit_phases()
        phase_mean = np.mean(phases)
        phase_std = np.std(phases)
        
        print(f"\nResults for n_bits={n_bits}:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        print(f"  Phase mean:     {phase_mean:.4f} (target: {math.pi:.4f})")
        print(f"  Phase std:      {phase_std:.4f}")
        
        results[f'holonomy_{n_bits}'] = {
            'n_bits': n_bits,
            'n_params': n_params,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'phase_mean': phase_mean,
            'phase_std': phase_std,
            'phases': phases.tolist(),
            'history': history
        }
        
        # Plot phase distribution
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(phases, bins=20, edgecolor='black')
        plt.axvline(x=math.pi, color='r', linestyle='--', label=f'pi={math.pi:.2f}')
        plt.xlabel('Phase (radians)')
        plt.ylabel('Count')
        plt.title(f'HolonomyChain n={n_bits}: Learned Phases')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'HolonomyChain n={n_bits}: Training Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'mnist_holonomy_{n_bits}_phases.png', dpi=150)
        plt.close()
        print(f"Saved plot to mnist_holonomy_{n_bits}_phases.png")
    
    # MLP baseline
    print(f"\n{'='*50}")
    print("Testing MLP Baseline")
    print(f"{'='*50}")
    
    # Match parameter count from n_bits=32
    mlp = StandardMLP(input_dim=input_dim)
    mlp_params = count_params(mlp)
    print(f"MLP params: {mlp_params}")
    
    # Train MLP
    print("\nTraining MLP...")
    mlp_history = train_mlp(mlp, train_loader, epochs=200, lr=0.01)
    
    # Evaluate
    mlp_test_acc = evaluate(mlp, test_loader)
    mlp_train_acc = evaluate(mlp, train_loader)
    
    print(f"\nMLP Results:")
    print(f"  Train accuracy: {mlp_train_acc:.4f}")
    print(f"  Test accuracy:  {mlp_test_acc:.4f}")
    
    results['mlp'] = {
        'n_params': mlp_params,
        'train_acc': mlp_train_acc,
        'test_acc': mlp_test_acc,
        'history': mlp_history
    }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n| Model            | Params | Train Acc | Test Acc |")
    print(f"|------------------|--------|------------|----------|")
    print(f"| Holonomy n=16    | {results['holonomy_16']['n_params']:6d} |   {results['holonomy_16']['train_acc']:.4f}   |  {results['holonomy_16']['test_acc']:.4f}  |")
    print(f"| Holonomy n=32    | {results['holonomy_32']['n_params']:6d} |   {results['holonomy_32']['train_acc']:.4f}   |  {results['holonomy_32']['test_acc']:.4f}  |")
    print(f"| MLP Baseline     | {mlp_params:6d} |   {mlp_train_acc:.4f}   |  {mlp_test_acc:.4f}  |")
    
    print("\n" + "-"*60)
    print("PHASE ANALYSIS:")
    print("-"*60)
    for n_bits in [16, 32]:
        r = results[f'holonomy_{n_bits}']
        print(f"n={n_bits}: mean={r['phase_mean']:.4f}, std={r['phase_std']:.4f}")
        print(f"  Phases: {[f'{p:.2f}' for p in r['phases'][:8]]}...")
    
    # Save results
    with open('mnist_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to mnist_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
