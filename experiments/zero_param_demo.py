import torch
import torch.nn as nn
import math

DEVICE = torch.device('cpu')

class ZkBundleExplicit(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        class_phases = 2 * math.pi * torch.arange(k, device=DEVICE) / k
        self.register_buffer('fourier_basis', torch.stack([
            torch.cos(class_phases), torch.sin(class_phases)
        ], dim=1))
    
    def forward(self, a, b):
        phases_a = 2 * math.pi * a.float() / self.k
        phases_b = 2 * math.pi * b.float() / self.k
        result_phase = phases_a + phases_b
        
        result_emb = torch.stack([
            torch.cos(result_phase),
            torch.sin(result_phase)
        ], dim=-1)
        
        logits = result_emb @ self.fourier_basis.T
        return logits

def make_data(k):
    a = torch.arange(k, device=DEVICE).repeat(k)
    b = torch.arange(k, device=DEVICE).repeat_interleave(k)
    target = (a + b) % k
    
    n = k * k
    indices = torch.randperm(n)
    train_idx = indices[:n // 2]
    test_idx = indices[n // 2:]
    
    return {
        'a_train': a[train_idx], 'b_train': b[train_idx], 'target_train': target[train_idx],
        'a_test': a[test_idx], 'b_test': b[test_idx], 'target_test': target[test_idx]
    }

print("="*60)
print("ZERO-PARAMETER GROKKING DEMONSTRATION")
print("="*60)

k = 23
print(f"\nTask: (a + b) mod {k}")

model = ZkBundleExplicit(k)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n1. LEARNABLE PARAMETERS: {num_params}")
print(f"   OPTIMIZATION STEPS: 0 (no optimizer needed)")

data = make_data(k)

with torch.no_grad():
    train_logits = model(data['a_train'], data['b_train'])
    train_acc = (train_logits.argmax(dim=-1) == data['target_train']).float().mean().item()
    
    test_logits = model(data['a_test'], data['b_test'])
    test_acc = (test_logits.argmax(dim=-1) == data['target_test']).float().mean().item()

print(f"\n2. ACCURACY AT STEP 0:")
print(f"   Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\n3. COMPARISON WITH GROKKING PAPERS:")
print(f"   - Power et al (2021): grokking at ~10,000-20,000 steps")
print(f"   - Gromov (2023): grokking at ~5,000-15,000 steps")
print(f"   - This work: grokking at STEP 0")

print(f"\n4. KEY INSIGHT:")
print(f"   The model DOES NOT LEARN any weights.")
print(f"   The geometry IS the solution.")
print(f"   Phase addition on circle = modular addition.")
print(f"   Fourier readout = perfect classifier.")
print(f"   Zero parameters, zero training, zero gradient steps.")

print("\n" + "="*60)
print("CONCLUSION: Grokking is not a 'magical phase transition'.")
print("It's the discovery of geometric structure from flat primitives.")
print("="*60)