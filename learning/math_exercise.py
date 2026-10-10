"""A three-class toy gradient check, not the IntentDesk neural model.

Run: uv run python learning/math_exercise.py
Try changing the learning rate or input vector and explain the result.
"""

import numpy as np


def loss_and_gradient(weights, features, target):
    logits = weights @ features
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    loss = np.log(np.exp(shifted).sum()) - shifted[target]
    dz = probabilities.copy()
    dz[target] -= 1
    gradient = np.outer(dz, features)
    return loss, gradient, probabilities


def main():
    rng = np.random.default_rng(42)
    weights = rng.normal(scale=0.1, size=(3, 4))
    features = np.array([1.0, 0.5, -0.2, 0.8])
    target = 1
    loss, gradient, probabilities = loss_and_gradient(weights, features, target)
    numerical = np.zeros_like(weights)
    epsilon = 1e-5
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            loss_and_gradient(plus, features, target)[0]
            - loss_and_gradient(minus, features, target)[0]
        ) / (2 * epsilon)
    error = np.max(np.abs(numerical - gradient))
    next_weights = weights - 0.1 * gradient
    next_loss = loss_and_gradient(next_weights, features, target)[0]
    assert error < 1e-6, "Analytic gradient disagrees with numerical estimate."
    assert next_loss < loss, "This example's SGD step should lower the loss."
    print("Toy input shape:", features.shape, "weight shape:", weights.shape)
    print("Initial class probabilities:", probabilities.round(4))
    print(f"Maximum gradient-check error: {error:.3e}")
    print(f"Cross-entropy before/after one SGD step: {loss:.6f} -> {next_loss:.6f}")


if __name__ == "__main__":
    main()

