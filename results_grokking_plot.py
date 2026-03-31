import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

with open('results_grokking.json', 'r') as f:
    results = json.load(f)

K_VALUES = [11, 17, 23]
SEEDS = [42, 123, 7]

Path('results').mkdir(exist_ok=True)

print("\n" + "="*70)
print("GROKKING DISCOVERY RESULTS")
print("="*70)

print("\n   k  | seed | grok_step | R²_before | R²_at_grok")
print("-"*50)

summary_data = []

for k in K_VALUES:
    for run in results['model_a'][str(k)]:
        seed = run['seed']
        grok_step = run['grokking_step']
        h = run['history']
        
        steps = h['step']
        align_data = h['fourier_alignment']
        
        align_dict = {x[0]: x[1] for x in align_data}
        
        r2_before_step = grok_step - 500
        if r2_before_step < 0:
            r2_before = align_dict.get(0, 0)
        else:
            before_keys = [k for k in align_dict.keys() if k <= r2_before_step]
            r2_before = align_dict[max(before_keys)] if before_keys else align_dict.get(0, 0)
        
        r2_at_grok = align_dict.get(grok_step, align_dict.get(steps[-1], 0))
        
        print(f"  {k:2d}  | {seed:3d}  |    {grok_step:5d}   |   {r2_before:.3f}   |   {r2_at_grok:.3f}")
        
        summary_data.append({'k': k, 'seed': seed, 'grok_step': grok_step, 'r2_before': r2_before, 'r2_at_grok': r2_at_grok})

print("-"*50)

print("\n   Model C (ZkBundleExplicit): 100% at step 0 for all k")

print("\n" + "="*70)
print("SUMMARY STATS")
print("="*70)

for k in K_VALUES:
    k_data = [d for d in summary_data if d['k'] == k]
    avg_grok = np.mean([d['grok_step'] for d in k_data])
    std_grok = np.std([d['grok_step'] for d in k_data])
    avg_r2_before = np.mean([d['r2_before'] for d in k_data])
    avg_r2_at = np.mean([d['r2_at_grok'] for d in k_data])
    print(f"k={k}: grok_step = {avg_grok:.0f} ± {std_grok:.0f}, R²_before = {avg_r2_before:.3f}, R²_at_grok = {avg_r2_at:.3f}")

print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
model_c_acc = results['model_c']

colors = {'42': 'blue', '123': 'green', '7': 'orange'}
alphas = {'42': 0.3, '123': 0.5, '7': 0.7}

for i, k in enumerate(K_VALUES):
    ax_acc = axes[i, 0]
    ax_align = axes[i, 1]
    
    for run in results['model_a'][str(k)]:
        h = run['history']
        seed_str = str(run['seed'])
        steps = h['step']
        
        ax_acc.plot(steps, h['test_acc'], color=colors[seed_str], alpha=alphas[seed_str], linewidth=1)
        ax_acc.plot(steps, h['train_acc'], color=colors[seed_str], linestyle='--', alpha=alphas[seed_str]/2, linewidth=1)
        
        align_steps = [x[0] for x in h['fourier_alignment']]
        align_vals = [x[1] for x in h['fourier_alignment']]
        ax_align.plot(align_steps, align_vals, color=colors[seed_str], alpha=alphas[seed_str], linewidth=1)
    
    h0 = results['model_a'][str(k)][0]['history']
    ax_acc.axhline(y=model_c_acc[str(k)]['test_acc'], color='red', linestyle='--', linewidth=2, label='ZkBundle (step 0)')
    
    mean_grok = np.mean([r['grokking_step'] for r in results['model_a'][str(k)]])
    ax_acc.axvline(x=mean_grok, color='green', linestyle=':', linewidth=2, label=f'grokking (mean={mean_grok:.0f})')
    ax_align.axvline(x=mean_grok, color='green', linestyle=':', linewidth=2)
    
    ax_acc.set_title(f'k={k} - Accuracy')
    ax_acc.set_xlabel('Steps')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)
    
    ax_align.set_title(f'k={k} - Fourier Alignment (R²)')
    ax_align.set_xlabel('Steps')
    ax_align.set_ylabel('R²')
    ax_align.set_ylim(0, 1.05)
    ax_align.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/grokking_discovery.png', dpi=150)
plt.close()
print("Saved results/grokking_discovery.png")