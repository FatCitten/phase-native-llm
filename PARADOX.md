# THE GROKKING PARADOX

## What Grokking Papers Define

Grokking is defined as "sudden generalization after prolonged overfitting" — the phenomenon where a neural network first memorizes training data, then suddenly generalizes to test data after extensive training (Power et al. 2021, Gromov 2023).

## What We Found

We present ZkBundleExplicit: a model that achieves 100% test accuracy on modular addition (a ≡ b (mod k)) with **zero learned parameters** and **zero gradient updates**. The architecture:
- Encodes inputs as points on the circle (phase = 2π·x/k)
- Performs angle addition (the group operation)
- Reads out via Fourier basis (the optimal classifier)

At initialization, this model already achieves perfect generalization. There is no "grokking" because there is nothing to discover — the geometric structure is built into the architecture.

## What This Means

We propose a new definition:

> **Grokking is not a phase transition — it is the cost of discovering geometric structure from flat primitives.**

When a network has flat embeddings (nn.Embedding), it must learn geometry through gradient descent. The "grokking step" measures how many gradient updates are needed to discover that modular addition corresponds to angle addition on a circle, and that the optimal classifier is the Fourier basis.

When the geometry is encoded structurally (as in ZkBundle), grokking is instantaneous — there is nothing to discover because the solution is already present.

## Conclusion

Grokking papers do not measure "magical generalization." They measure the computational cost of structure discovery. Our zero-parameter solution demonstrates that the task was always solvable — the network just needed the right primitives.