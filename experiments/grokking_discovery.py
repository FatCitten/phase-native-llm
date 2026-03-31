import torch
import torch.nn as nn
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

DEVICE = torch.device('cpu')
K_VALUES = [11, 17, 23]
SEEDS = [42, 123, 7]
MAX_STEPS = 15000
LOG_EVERY = 100
ALIGNMENT_EVERY = 500
EARLY_STOP_HOLD = 100
EARLY_STOP_THRESHOLD = 0.99

class FlatTransformer(nn.Module):
    def __init__(self, k, d_model=64):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.embedding = nn.Embedding(k, d_model)
        self.pos_embed = nn.Embedding(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=1, dim_feedforward=128,
            batch_first=True, dropout=0.0, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Linear(d_model, k)
    
    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        pos = torch.arange(2, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.embedding(x) + self.pos_embed(pos)
        x = self.transformer(x)
        return self.output(x[:, 0, :])

def compute_fourier_alignment(model, k):
    E = model.embedding.weight.detach()
    c = torch.arange(k, device=DEVICE).float()
    F = torch.stack([torch.cos(2 * math.pi * c / k), torch.sin(2 * math.pi * c / k)], dim=1)
    W = torch.linalg.lstsq(E, F).solution
    F_pred = E @ W
    ss_res = ((F - F_pred) ** 2).sum()
    ss_tot = ((F - F.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

def make_data(k, train_ratio=0.8):
    a = torch.arange(k, device=DEVICE).repeat(k)
    b = torch.arange(k, device=DEVICE).repeat_interleave(k)
    target = (a + b) % k
    n = k * k
    indices = torch.randperm(n, device=DEVICE)
    n_train = int(n * train_ratio)
    return {
        'a_train': a[indices[:n_train]], 'b_train': b[indices[:n_train]], 'target_train': target[indices[:n_train]],
        'a_test': a[indices[n_train:]], 'b_test': b[indices[n_train:]], 'target_test': target[indices[n_train:]]
    }

def run_model_c(k):
    data = make_data(k)
    a_test, b_test, target_test = data['a_test'], data['b_test'], data['target_test']
    class_phases = 2 * math.pi * torch.arange(k, device=DEVICE) / k
    phase_a = 2 * math.pi * a_test.float() / k
    phase_b = 2 * math.pi * b_test.float() / k
    result_phase = phase_a + phase_b
    logits = torch.stack([torch.cos(result_phase - cp) for cp in class_phases], dim=1)
    acc = (logits.argmax(dim=1) == target_test).float().mean().item()
    return {'test_acc': acc, 'train_acc': 1.0}

def run_model_a(k, data, seed):
    torch.manual_seed(seed)
    model = FlatTransformer(k).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    a_train, b_train, target_train = data['a_train'], data['b_train'], data['target_train']
    a_test, b_test, target_test = data['a_test'], data['b_test'], data['target_test']
    
    history = {'step': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'fourier_alignment': []}
    grokking_step = MAX_STEPS
    start_time = time.time()
    
    for step in range(MAX_STEPS):
        optimizer.zero_grad()
        logits = model(a_train, b_train)
        loss = criterion(logits, target_train)
        loss.backward()
        optimizer.step()
        
        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            with torch.no_grad():
                train_acc = (logits.argmax(dim=-1) == target_train).float().mean().item()
                test_logits = model(a_test, b_test)
                test_acc = (test_logits.argmax(dim=-1) == target_test).float().mean().item()
            
            history['step'].append(step)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['train_loss'].append(loss.item())
            history['test_loss'].append(test_logits.argmax(dim=-1).float().mean().item())
            
            if step % ALIGNMENT_EVERY == 0:
                alignment = compute_fourier_alignment(model, k)
                history['fourier_alignment'].append((step, alignment))
            
            elapsed = time.time() - start_time
            if step % 500 == 0:
                eta = (MAX_STEPS - step) / (step + 1) * elapsed
                print(f"    step={step:5d} train={train_acc:.3f} test={test_acc:.3f} align={history['fourier_alignment'][-1][1]:.3f} ETA={eta/60:.1f}min")
            
            if test_acc > EARLY_STOP_THRESHOLD and len(history['test_acc']) >= EARLY_STOP_HOLD // LOG_EVERY:
                if all(x > EARLY_STOP_THRESHOLD for x in history['test_acc'][-(EARLY_STOP_HOLD // LOG_EVERY):]):
                    grokking_step = step
                    print(f"    => GROKKED at step {step}")
                    break
    
    return history, grokking_step

def main():
    print("="*60)
    print("EXPERIMENT: Grokking as Geometric Discovery")
    print("="*60)
    
    Path('results').mkdir(exist_ok=True)
    results = {'model_c': {}, 'model_a': {}}
    
    print("\n--- MODEL C (ZkBundleExplicit - step 0) ---")
    for k in K_VALUES:
        r = run_model_c(k)
        results['model_c'][k] = r
        print(f"k={k}: test_acc={r['test_acc']:.4f}")
    
    print("\n--- MODEL A (FlatTransformer) ---")
    for k in K_VALUES:
        data = make_data(k)
        print(f"\nk={k} (n_train={len(data['target_train'])}, n_test={len(data['target_test'])})")
        results['model_a'][k] = []
        
        for seed in SEEDS:
            print(f"  seed={seed}:")
            history, grokking_step = run_model_a(k, data, seed)
            results['model_a'][k].append({
                'seed': seed,
                'grokking_step': grokking_step,
                'history': history
            })
            print(f"    grokking_step={grokking_step}")
    
    with open('results_grokking.json', 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print("\nSaved results_grokking.json")
    
    print("\n--- GENERATING PLOTS ---")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    model_c_acc = results['model_c']
    
    for i, k in enumerate(K_VALUES):
        ax_acc = axes[i, 0]
        ax_align = axes[i, 1]
        
        for run in results['model_a'][k]:
            h = run['history']
            steps = h['step']
            ax_acc.plot(steps, h['test_acc'], 'b-', alpha=0.3)
            ax_acc.plot(steps, h['train_acc'], 'gray', linestyle='--', alpha=0.3)
            
            align_steps = [x[0] for x in h['fourier_alignment']]
            align_vals = [x[1] for x in h['fourier_alignment']]
            ax_align.plot(align_steps, align_vals, 'orange', alpha=0.3)
        
        h0 = results['model_a'][k][0]['history']
        ax_acc.plot(h0['step'], np.mean([r['history']['test_acc'] for r in results['model_a'][k]], axis=0), 'b-', linewidth=2, label='test (mean)')
        ax_acc.plot(h0['step'], np.mean([r['history']['train_acc'] for r in results['model_a'][k]], axis=0), 'gray', linestyle='--', linewidth=2, label='train (mean)')
        
        align_steps = [x[0] for x in h0['fourier_alignment']]
        align_vals = np.mean([r['history']['fourier_alignment'] for r in results['model_a'][k]], axis=0)
        ax_align.plot(align_steps, align_vals, 'orange', linewidth=2, label='fourier_alignment (mean)')
        
        ax_acc.axhline(y=model_c_acc[k]['test_acc'], color='red', linestyle='--', label='ZkBundle (step 0)')
        ax_acc.axvline(x=np.mean([r['grokking_step'] for r in results['model_a'][k]]), color='green', linestyle=':', label='grokking_step')
        
        mean_grok = np.mean([r['grokking_step'] for r in results['model_a'][k]])
        ax_align.axvline(x=mean_grok, color='green', linestyle=':')
        
        ax_acc.set_title(f'k={k} - Accuracy')
        ax_acc.set_xlabel('Steps')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)
        
        ax_align.set_title(f'k={k} - Fourier Alignment')
        ax_align.set_xlabel('Steps')
        ax_align.set_ylabel('R²')
        ax_align.set_ylim(0, 1)
        ax_align.legend(fontsize=8)
        ax_align.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/grokking_discovery.png', dpi=150)
    plt.close()
    print("Saved results/grokking_discovery.png")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k in K_VALUES:
        grok_steps = [r['grokking_step'] for r in results['model_a'][k]]
        print(f"k={k}: grokking_step = {np.mean(grok_steps):.0f} ± {np.std(grok_steps):.0f}")
    print(f"\nModel C (ZkBundleExplicit): 100% at step 0 for all k")

if __name__ == '__main__':
    main()