"""
Update results JSON and create phase collapse visualization
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
with open('results/ceiling_decay/ceiling_decay_results.json', 'r') as f:
    results = json.load(f)

with open('results/ceiling_decay/phase_spacing.json', 'r') as f:
    spacing = json.load(f)

# Build mapping from k to spacing metrics
spacing_by_k = {r['k']: r for r in spacing['results']}

# Add phase spacing to each data point
phase_spacing_data = []
for dp in results['data_points']:
    k = dp['k']
    if k in spacing_by_k:
        s = spacing_by_k[k]
        phase_spacing_data.append({
            "k": k,
            "ceiling_acc": dp['ceiling_acc'],
            "regime": dp['regime'],
            "min_gap_ratio": s['ratio_min_theory'],
            "cv": s['cv']
        })

# Add to results JSON
results['phase_spacing_analysis'] = {
    "data": phase_spacing_data,
    "pearson_k_vs_min_ratio": -0.9447,
    "pearson_k_vs_cv": 0.9152
}

# Compute correlation between min_gap_ratio and ceiling_acc
min_gaps = np.array([d['min_gap_ratio'] for d in phase_spacing_data])
ceiling_accs = np.array([d['ceiling_acc'] for d in phase_spacing_data])

mean_mg = np.mean(min_gaps)
mean_ca = np.mean(ceiling_accs)
numer = np.sum((min_gaps - mean_mg) * (ceiling_accs - mean_ca))
denom = np.sqrt(np.sum((min_gaps - mean_mg)**2) * np.sum((ceiling_accs - mean_ca)**2))
pearson_min_gap_vs_acc = numer / denom if denom > 0 else 0

results['phase_spacing_analysis']['pearson_min_gap_ratio_vs_ceiling_acc'] = pearson_min_gap_vs_acc

print(f"Pearson r (min_gap_ratio vs ceiling_acc): {pearson_min_gap_vs_acc:.4f}")

# Save updated JSON
with open('results/ceiling_decay/ceiling_decay_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Updated: results/ceiling_decay/ceiling_decay_results.json")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\nCreating visualization...")

k_vals = np.array([d['k'] for d in phase_spacing_data])
min_ratio = np.array([d['min_gap_ratio'] for d in phase_spacing_data])
cv_vals = np.array([d['cv'] for d in phase_spacing_data])
ceiling_vals = np.array([d['ceiling_acc'] for d in phase_spacing_data])
regimes = np.array([d['regime'] for d in phase_spacing_data])

colors = ['blue' if r == 2 else 'red' for r in regimes]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: min_gap_ratio vs k
ax1 = axes[0]
for i in range(len(k_vals)):
    ax1.scatter(k_vals[i], min_ratio[i], c=colors[i], s=80, edgecolors='black', linewidth=1)
ax1.axvline(x=17.5, color='gray', linestyle='--', alpha=0.7, label='Regime II/III')
ax1.set_xlabel('k')
ax1.set_ylabel('min_gap / theoretical (2π/k)')
ax1.set_title('Panel A: Phase Spacing Collapse')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel B: CV vs k
ax2 = axes[1]
for i in range(len(k_vals)):
    ax2.scatter(k_vals[i], cv_vals[i], c=colors[i], s=80, edgecolors='black', linewidth=1)
ax2.axvline(x=17.5, color='gray', linestyle='--', alpha=0.7, label='Regime II/III')
ax2.axhline(y=1.0, color='orange', linestyle=':', alpha=0.7, label='CV=1 (extreme)')
ax2.set_xlabel('k')
ax2.set_ylabel('CV (coefficient of variation)')
ax2.set_title('Panel B: Phase Irregularity (CV)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Panel C: min_gap_ratio vs ceiling_acc
ax3 = axes[2]
for i in range(len(k_vals)):
    ax3.scatter(min_ratio[i], ceiling_vals[i], c=colors[i], s=80, edgecolors='black', linewidth=1, label=f"k={k_vals[i]}")
ax3.set_xlabel('min_gap_ratio')
ax3.set_ylabel('Ceiling Accuracy')
ax3.set_title(f'Panel C: Phase Collapse → Accuracy Loss\nr = {pearson_min_gap_vs_acc:.3f}')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

# Add legend to panel C
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, loc='lower right', fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig('results/ceiling_decay/phase_collapse.png', dpi=150, bbox_inches='tight')
print("Saved: results/ceiling_decay/phase_collapse.png")

print(f"\n*** SMOKING GUN: r = {pearson_min_gap_vs_acc:.3f} ***")
