# Minimal Handoff for Next Agent

**Project:** Phase-native neural networks - geometric phase encoding

**Confirmed:**
- CRT works when gcd=1, fails when gcd>1
- Z_k works for **prime-order** k (Z_2, Z_3, Z_5, Z_7)
- Z_k fails for **prime-power** k (Z_4=2^2, Z_8=2^3, Z_9=3^2)
- **Z_6 untested** - interesting case (composite, NOT prime power)
- sigma*xk may NOT be constant - data fits sigma*xk = 1.944 - 0.623/k

**Run these 4 experiments in order:**
1. **NEW-B**: Fresh Z_4 run to reconfirm failure (10 seeds)
2. **NEW-A**: Phase convergence check (phases = 2*pi*j/k +/- offset)
3. **NEW-C**: Control with max(a,b) mod k
4. **NEW-D**: Distinguish between:
   - H1: sigma*xk = constant (~1.77-1.94)
   - H2: sigma*xk = 1.944 - 0.623/k (NOT constant)
   - Test k=3,5,7,11,13,17,19,23,29, 200 seeds each
   - **Report raw C(k) per k** - let data decide

**Output:** Raw numbers only.

**Code template provided in handoff** - agent doesn't need to read existing experiment files.

---

## Code Template

```python
import math, torch, torch.nn as nn, numpy as np

class ZkBundle(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
    
    def forward(self, x1, x2):
        p1, p2 = self.input_phases[x1], self.input_phases[x2]
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = torch.min(dists % (2 * math.pi), 2 * math.pi - (dists % (2 * math.pi)))
        return -dists
```

---

## Existing Results (don't need to re-run)
- experiment_1_results.json - sigma*xk data (50 seeds, k=3,5,7,11)
- experiment_b_results.json - CRT confirmation
- experiment_2_refined.py - Z_4 failure evidence

---
END
