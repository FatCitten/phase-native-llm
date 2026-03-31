"""
Ceiling Decay Law Analysis
==========================
Fit three models to ceiling_acc vs k data,
test for k=21 outlier, check antipodal bounds.
"""

import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# STEP 1: DATA (hardcoded exactly as specified)
# ============================================================================
data = [
    {"k": 5,  "ceiling_acc": 1.000, "N_est": 50, "regime": 2},
    {"k": 7,  "ceiling_acc": 1.000, "N_est": 40, "regime": 2},
    {"k": 11, "ceiling_acc": 1.000, "N_est": 22, "regime": 2},
    {"k": 19, "ceiling_acc": 0.842, "N_est": 20, "regime": 3},
    {"k": 21, "ceiling_acc": 0.952, "N_est": 20, "regime": 3},
    {"k": 23, "ceiling_acc": 0.826, "N_est": 22, "regime": 3},
    {"k": 29, "ceiling_acc": 0.690, "N_est": 12, "regime": 3},
]

k_vals = np.array([d["k"] for d in data])
ceiling_vals = np.array([d["ceiling_acc"] for d in data])
N_vals = np.array([d["N_est"] for d in data])
regime_vals = np.array([d["regime"] for d in data])
weights = np.sqrt(N_vals)

# Separate Regime III points
regime3_mask = regime_vals == 3
k3 = k_vals[regime3_mask]
ceiling3 = ceiling_vals[regime3_mask]
N3 = N_vals[regime3_mask]

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
def model_A_exp(k, tau, beta):
    """Stretched exponential from critical point k*=17"""
    k = np.asarray(k)
    result = np.where(k > 17, np.exp(-((k - 17) / tau) ** beta), 1.0)
    return result

def model_B_power(k, k_star, alpha):
    """Power law ceiling"""
    k = np.asarray(k)
    result = np.minimum(1.0, (k_star / k) ** alpha)
    return result

def model_C_logistic(k, gamma, k_mid):
    """Logistic drop"""
    k = np.asarray(k)
    return 1.0 / (1.0 + np.exp(gamma * (k - k_mid)))

def fit_model(model_func, k_data, y_data, weights, p0, bounds):
    """Fit model and compute metrics"""
    try:
        popt, pcov = curve_fit(model_func, k_data, y_data, p0=p0, bounds=bounds, 
                               sigma=weights, absolute_sigma=True)
        y_pred = model_func(k_data, *popt)
        
        ss_res = np.sum(weights**2 * (y_data - y_pred)**2)
        ss_tot = np.sum(weights**2 * (y_data - np.average(y_data, weights=weights))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rmse = np.sqrt(np.mean((y_data - y_pred)**2))
        
        # 95% CI from covariance
        perr = np.sqrt(np.diag(pcov))
        ci_lower = popt - 1.96 * perr
        ci_upper = popt + 1.96 * perr
        
        return {
            "params": list(popt),
            "param_names": ["tau", "beta"] if model_func == model_A_exp else 
                          ["k_star", "alpha"] if model_func == model_B_power else ["gamma", "k_mid"],
            "ci_lower": list(ci_lower),
            "ci_upper": list(ci_upper),
            "r2": float(r2),
            "rmse": float(rmse),
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

# ============================================================================
# FIT ALL MODELS (with k=21)
# ============================================================================
print("=" * 60)
print("FITTING MODELS WITH ALL 7 POINTS (including k=21)")
print("=" * 60)

fit_A_with = fit_model(model_A_exp, k_vals, ceiling_vals, weights, [5.0, 1.5], ([1, 0.1], [50, 10]))
fit_B_with = fit_model(model_B_power, k_vals, ceiling_vals, weights, [17.0, 3.0], ([1, 0.1], [50, 20]))
fit_C_with = fit_model(model_C_logistic, k_vals, ceiling_vals, weights, [0.3, 22.0], ([0.01, 5], [5, 40]))

print(f"\nModel A (exponential): tau={fit_A_with['params'][0]:.3f}+/-{fit_A_with['ci_upper'][0]-fit_A_with['params'][0]:.3f}, "
      f"beta={fit_A_with['params'][1]:.3f}+/-{fit_A_with['ci_upper'][1]-fit_A_with['params'][1]:.3f}, R2={fit_A_with['r2']:.4f}")
print(f"Model B (power law): k*={fit_B_with['params'][0]:.3f}+/-{fit_B_with['ci_upper'][0]-fit_B_with['params'][0]:.3f}, "
      f"alpha={fit_B_with['params'][1]:.3f}+/-{fit_B_with['ci_upper'][1]-fit_B_with['params'][1]:.3f}, R2={fit_B_with['r2']:.4f}")
print(f"Model C (logistic): gamma={fit_C_with['params'][0]:.3f}+/-{fit_C_with['ci_upper'][0]-fit_C_with['params'][0]:.3f}, "
      f"k_mid={fit_C_with['params'][1]:.3f}+/-{fit_C_with['ci_upper'][1]-fit_C_with['params'][1]:.3f}, R2={fit_C_with['r2']:.4f}")

# ============================================================================
# FIT ALL MODELS (without k=21)
# ============================================================================
mask_no_k21 = k_vals != 21
k_no21 = k_vals[mask_no_k21]
ceiling_no21 = ceiling_vals[mask_no_k21]
N_no21 = N_vals[mask_no_k21]
weights_no21 = np.sqrt(N_no21)

print("\n" + "=" * 60)
print("FITTING MODELS WITHOUT k=21 (6 points)")
print("=" * 60)

fit_A_without = fit_model(model_A_exp, k_no21, ceiling_no21, weights_no21, [5.0, 1.5], ([1, 0.1], [50, 10]))
fit_B_without = fit_model(model_B_power, k_no21, ceiling_no21, weights_no21, [17.0, 3.0], ([1, 0.1], [50, 20]))
fit_C_without = fit_model(model_C_logistic, k_no21, ceiling_no21, weights_no21, [0.3, 22.0], ([0.01, 5], [5, 40]))

print(f"\nModel A (exponential): tau={fit_A_without['params'][0]:.3f}+/-{fit_A_without['ci_upper'][0]-fit_A_without['params'][0]:.3f}, "
      f"beta={fit_A_without['params'][1]:.3f}+/-{fit_A_without['ci_upper'][1]-fit_A_without['params'][1]:.3f}, R2={fit_A_without['r2']:.4f}")
print(f"Model B (power law): k*={fit_B_without['params'][0]:.3f}+/-{fit_B_without['ci_upper'][0]-fit_B_without['params'][0]:.3f}, "
      f"alpha={fit_B_without['params'][1]:.3f}+/-{fit_B_without['ci_upper'][1]-fit_B_without['params'][1]:.3f}, R2={fit_B_without['r2']:.4f}")
print(f"Model C (logistic): gamma={fit_C_without['params'][0]:.3f}+/-{fit_C_without['ci_upper'][0]-fit_C_without['params'][0]:.3f}, "
      f"k_mid={fit_C_without['params'][1]:.3f}+/-{fit_C_without['ci_upper'][1]-fit_C_without['params'][1]:.3f}, R2={fit_C_without['r2']:.4f}")

# ============================================================================
# STEP 2: LOG-LOG LINEARITY TEST
# ============================================================================
print("\n" + "=" * 60)
print("LOG-LOG LINEARITY TEST (Regime III only)")
print("=" * 60)

# With k=21
log_k3 = np.log(k3)
log_ceiling3 = np.log(ceiling3)
m_with, c_with = np.polyfit(log_k3, log_ceiling3, 1)
log_pred_with = m_with * log_k3 + c_with
ss_res_ll_with = np.sum((log_ceiling3 - log_pred_with)**2)
ss_tot_ll_with = np.sum((log_ceiling3 - np.mean(log_ceiling3))**2)
r2_ll_with = 1 - ss_res_ll_with / ss_tot_ll_with

print(f"\nWith k=21: slope={m_with:.4f}, R2={r2_ll_with:.4f}")
verdict_with = "POWER LAW CONFIRMED (log-log)" if r2_ll_with > 0.95 else "POWER LAW WEAK (log-log)"
print(verdict_with)

# Without k=21
k3_no21_mask = k3 != 21
k3_no21 = k3[k3_no21_mask]
ceiling3_no21 = ceiling3[k3_no21_mask]
log_k3_no21 = np.log(k3_no21)
log_ceiling3_no21 = np.log(ceiling3_no21)
m_without, c_without = np.polyfit(log_k3_no21, log_ceiling3_no21, 1)
log_pred_without = m_without * log_k3_no21 + c_without
ss_res_ll_without = np.sum((log_ceiling3_no21 - log_pred_without)**2)
ss_tot_ll_without = np.sum((log_ceiling3_no21 - np.mean(log_ceiling3_no21))**2)
r2_ll_without = 1 - ss_res_ll_without / ss_tot_ll_without

print(f"\nWithout k=21: slope={m_without:.4f}, R2={r2_ll_without:.4f}")
verdict_without = "POWER LAW CONFIRMED (log-log)" if r2_ll_without > 0.95 else "POWER LAW WEAK (log-log)"
print(verdict_without)

# ============================================================================
# STEP 3: ANTIPODAL CHECK
# ============================================================================
print("\n" + "=" * 60)
print("ANTIPODAL BOUND CHECK")
print("=" * 60)
print(f"\n{'k':>4} | {'ceiling':>8} | {'antipodal_bound':>15} | {'gap':>8} | {'below?':>7}")
print("-" * 55)

antipodal_results = []
below_count = 0
for d in data:
    k = d["k"]
    ceiling = d["ceiling_acc"]
    bound = (k - 1) / k
    gap = ceiling - bound
    below = gap < 0
    if below:
        below_count += 1
    antipodal_results.append({
        "k": k, "ceiling_acc": ceiling, "antipodal_bound": bound, 
        "gap": gap, "below_bound": below
    })
    print(f"{k:>4} | {ceiling:>8.3f} | {bound:>15.3f} | {gap:>8.3f} | {'YES' if below else 'NO':>7}")

# ============================================================================
# STEP 4: k=21 OUTLIER VERDICT
# ============================================================================
print("\n" + "=" * 60)
print("k=21 OUTLIER ANALYSIS")
print("=" * 60)

delta_r2_A = fit_A_with['r2'] - fit_A_without['r2']
delta_r2_B = fit_B_with['r2'] - fit_B_without['r2']
delta_r2_C = fit_C_with['r2'] - fit_C_without['r2']

print(f"\nDelta R2 with vs without k=21:")
print(f"  Model A: {delta_r2_A:+.4f}")
print(f"  Model B: {delta_r2_B:+.4f}")
print(f"  Model C: {delta_r2_C:+.4f}")

max_delta = max(delta_r2_A, delta_r2_B, delta_r2_C)
k21_outlier = max_delta > 0.10

if k21_outlier:
    print(f"\nk=21 IS A GENUINE OUTLIER (max delta={max_delta:.4f} > 0.10)")
else:
    print(f"\nk=21 is within noise (max delta={max_delta:.4f} <= 0.10)")

# ============================================================================
# STEP 5: PLOT
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING PLOT")
print("=" * 60)

Path("results/ceiling_decay").mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Common setup
k_fit = np.linspace(5, 31, 200)

# Error bars (binomial SE)
err = 1.0 / np.sqrt(N_vals)

# Colors
colors = ['blue' if r == 2 else 'red' for r in regime_vals]

# Panel A: Linear scale
ax1 = axes[0]
for i, d in enumerate(data):
    marker = 'o' if d['k'] != 21 else 'o'
    facecolor = 'none' if d['k'] == 21 else colors[i]
    edgecolor = 'red' if d['k'] == 21 else colors[i]
    ax1.errorbar(d['k'], d['ceiling_acc'], yerr=err[i], 
                 fmt=marker, markersize=8, color=edgecolor, 
                 markerfacecolor=facecolor, markeredgewidth=2,
                 capsize=3, label=f"k={d['k']}" if i == 0 else "")
    if d['k'] == 21:
        ax1.annotate("N~20", (d['k'], d['ceiling_acc']+0.08), ha='center', fontsize=8)

# Model curves (with k=21 fits)
k_fit_A = np.linspace(17, 31, 100)
ax1.plot(k_fit_A, model_A_exp(k_fit_A, *fit_A_with['params']), 'g--', 
         label=f"A (exp): R2={fit_A_with['r2']:.3f}", alpha=0.7)
ax1.plot(k_fit, model_B_power(k_fit, *fit_B_with['params']), 'b--', 
         label=f"B (power): R2={fit_B_with['r2']:.3f}", alpha=0.7)
ax1.plot(k_fit, model_C_logistic(k_fit, *fit_C_with['params']), 'm--', 
         label=f"C (logistic): R2={fit_C_with['r2']:.3f}", alpha=0.7)

ax1.axvline(x=17, color='gray', linestyle=':', alpha=0.7, label="Regime boundary")
ax1.axvline(x=18.5, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('k')
ax1.set_ylabel('Ceiling Accuracy')
ax1.set_title('Ceiling Decay vs k (Linear)')
ax1.set_xlim(3, 32)
ax1.set_ylim(0.5, 1.1)
ax1.legend(loc='lower left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel B: Log-log scale
ax2 = axes[1]
for i, d in enumerate(data):
    marker = 'o' if d['k'] != 21 else 'o'
    facecolor = 'none' if d['k'] == 21 else colors[i]
    edgecolor = 'red' if d['k'] == 21 else colors[i]
    ax2.errorbar(d['k'], d['ceiling_acc'], yerr=err[i], 
                 fmt=marker, markersize=8, color=edgecolor, 
                 markerfacecolor=facecolor, markeredgewidth=2,
                 capsize=3, label=f"k={d['k']}" if i == 0 else "")
    if d['k'] == 21:
        ax2.annotate("N~20", (d['k'], d['ceiling_acc']+0.08), ha='center', fontsize=8)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('k (log scale)')
ax2.set_ylabel('Ceiling Accuracy (log scale)')
ax2.set_title('Ceiling Decay vs k (Log-Log)')
ax2.set_xlim(4, 35)
ax2.set_ylim(0.5, 1.1)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig("results/ceiling_decay/ceiling_decay_law.png", dpi=150, bbox_inches='tight')
print("Saved: results/ceiling_decay/ceiling_decay_law.png")

# ============================================================================
# STEP 6: SAVE JSON
# ============================================================================
print("\n" + "=" * 60)
print("SAVING JSON RESULTS")
print("=" * 60)

# Determine winner (without k=21)
fits_r2 = {"A": fit_A_without['r2'], "B": fit_B_without['r2'], "C": fit_C_without['r2']}
winner = max(fits_r2, key=fits_r2.get)
winner_name = {"A": "exponential", "B": "power_law", "C": "logistic"}[winner]

# Headline based on winner
if winner == "A":
    headline = f"Ceiling decays as stretched exponential with tau={fit_A_without['params'][0]:.2f}, beta={fit_A_without['params'][1]:.2f}"
elif winner == "B":
    headline = f"Ceiling follows power law: ceiling ~ (k*/k)^alpha with k*={fit_B_without['params'][0]:.1f}, alpha={fit_B_without['params'][1]:.2f}"
else:
    headline = f"Ceiling drops logistically at k_mid={fit_C_without['params'][1]:.1f}"

results_json = {
    "data_points": data,
    "fits": {
        "with_k21": {
            "model_A": fit_A_with,
            "model_B": fit_B_with,
            "model_C": fit_C_with
        },
        "without_k21": {
            "model_A": fit_A_without,
            "model_B": fit_B_without,
            "model_C": fit_C_without
        }
    },
    "loglog_test": {
        "with_k21": {"slope": float(m_with), "r2": float(r2_ll_with), 
                     "verdict": verdict_with},
        "without_k21": {"slope": float(m_without), "r2": float(r2_ll_without),
                       "verdict": verdict_without}
    },
    "antipodal_check": antipodal_results,
    "k21_outlier": {
        "delta_r2_A": float(delta_r2_A),
        "delta_r2_B": float(delta_r2_B),
        "delta_r2_C": float(delta_r2_C),
        "verdict": "OUTLIER" if k21_outlier else "NOT_OUTLIER"
    },
    "winner": winner_name,
    "headline_finding": headline
}

with open("results/ceiling_decay/ceiling_decay_results.json", 'w') as f:
    json.dump(results_json, f, indent=2)
print("Saved: results/ceiling_decay/ceiling_decay_results.json")

# ============================================================================
# STEP 7: PRINT VERDICT
# ============================================================================
print("\n" + "=" * 60)
print("CEILING DECAY LAW VERDICT")
print("=" * 60)

# Format parameters based on winner
if winner == "A":
    params_str = f"tau={fit_A_without['params'][0]:.2f}, beta={fit_A_without['params'][1]:.2f}"
elif winner == "B":
    params_str = f"k*={fit_B_without['params'][0]:.1f}, alpha={fit_B_without['params'][1]:.2f}"
else:
    params_str = f"gamma={fit_C_without['params'][0]:.2f}, k_mid={fit_C_without['params'][1]:.1f}"

print("""
  +================================================+
  |        CEILING DECAY LAW VERDICT               |
  +================================================+
  | Best model:    {winner}/{winner_name}                           |
  | Parameters:    {params_str}          |
  | R2 (w/ k21):  {r2_with_all}                   |
  | R2 (w/o k21): {r2_without_all}                   |
  | Log-log R2:   {r2_ll:.3f} -> {verdict_ll}                          |
  | k=21 outlier: {outlier}                                    |
  | Antipodal:    {below_count}/7 points below bound                  |
  +================================================+
  | HEADLINE:                                      |
  | {headline}           |
  +================================================+
""".format(
    winner=winner,
    winner_name=winner_name,
    params_str=params_str,
    r2_with_all=f"{fit_A_with['r2']:.3f} / {fit_B_with['r2']:.3f} / {fit_C_with['r2']:.3f}",
    r2_without_all=f"{fit_A_without['r2']:.3f} / {fit_B_without['r2']:.3f} / {fit_C_without['r2']:.3f}",
    r2_ll=r2_ll_with,
    verdict_ll=verdict_with[:10],
    outlier="YES" if k21_outlier else "NO",
    below_count=below_count,
    headline=headline[:50]
))

print("\nDone!")
