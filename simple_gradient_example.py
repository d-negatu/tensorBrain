#!/usr/bin/env python3
"""
Simple gradient example to understand how gradients work!
This will teach you gradients by seeing them in action.
"""

import numpy as np

# Let's start with a super simple version without our Tensor class
# We'll build the Tensor class step by step

print("🧠 Understanding Gradients with Simple Examples")
print("=" * 50)

# Example 1: Simple function y = 2x
print("\n📚 Example 1: y = 2x")
print("If x = 5, then y = 2 * 5 = 10")
print("The gradient dy/dx = 2")
print("This means: if x increases by 1, y increases by 2")

x = 5.0
y = 2 * x
gradient = 2  # We know this from calculus, but let's see it in action

print(f"\nCurrent x: {x}")
print(f"Current y: {y}")
print(f"Gradient: {gradient}")

# Example 2: Loss function
print("\n📚 Example 2: Loss = (y - target)²")
print("If y = 10 and target = 12, then loss = (10 - 12)² = 4")
print("The gradient d(loss)/dy = 2 * (y - target) = 2 * (10 - 12) = -4")

target = 12.0
loss = (y - target) ** 2
loss_gradient = 2 * (y - target)

print(f"\nCurrent y: {y}")
print(f"Target: {target}")
print(f"Loss: {loss}")
print(f"Loss gradient: {loss_gradient}")

# Example 3: Chain rule - gradient of loss w.r.t. x
print("\n📚 Example 3: Chain Rule")
print("We want: d(loss)/dx")
print("Using chain rule: d(loss)/dx = d(loss)/dy * dy/dx")
print("d(loss)/dx = -4 * 2 = -8")

chain_gradient = loss_gradient * gradient

print(f"\nChain rule result: {chain_gradient}")
print(f"This means: if x increases by 1, loss increases by {chain_gradient}")

# Example 4: Gradient descent - using gradient to improve
print("\n📚 Example 4: Gradient Descent")
print("We want to minimize the loss")
print("Since gradient is negative, we should INCREASE x")
print("New x = x - learning_rate * gradient")

learning_rate = 0.1
new_x = x - learning_rate * chain_gradient
new_y = 2 * new_x
new_loss = (new_y - target) ** 2

print(f"\nOld x: {x}")
print(f"Old loss: {loss}")
print(f"New x: {new_x}")
print(f"New y: {new_y}")
print(f"New loss: {new_loss}")
print(f"Improvement: {loss - new_loss}")

print("\n🎉 Key Insights:")
print("1. Gradient tells us direction: negative = increase, positive = decrease")
print("2. Gradient tells us magnitude: bigger = make bigger change")
print("3. We use gradient to update parameters: x = x - lr * gradient")
print("4. This is how neural networks learn!")

print("\n🚀 Next: We'll build this into our Tensor class!")
print("Then you can do: x = Tensor(5.0, requires_grad=True)")
print("And the computer will compute gradients automatically!")
