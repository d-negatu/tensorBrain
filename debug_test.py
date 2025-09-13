#!/usr/bin/env python3
"""
Debug test to see what's happening with gradients
"""

from tensor import Tensor
import numpy as np

print("🔍 Debugging gradient issue...")

# Test scalar tensor
print("\n📝 Test 1: Scalar tensor")
x = Tensor(5.0, requires_grad=True)
print(f"Created: {x}")
print(f"Initial grad: {x.grad}")

x.backward()
print(f"After backward: grad = {x.grad}")
print(f"Grad type: {type(x.grad)}")
print(f"Grad shape: {x.grad.shape if hasattr(x.grad, 'shape') else 'No shape'}")

# Test vector tensor
print("\n📝 Test 2: Vector tensor")
y = Tensor([1, 2, 3], requires_grad=True)
print(f"Created: {y}")
print(f"Initial grad: {y.grad}")

gradient = np.array([0.1, 0.2, 0.3])
print(f"Gradient to pass: {gradient}")
print(f"Gradient type: {type(gradient)}")
print(f"Gradient shape: {gradient.shape}")

y.backward(gradient)
print(f"After backward: grad = {y.grad}")
print(f"Grad type: {type(y.grad)}")
print(f"Grad shape: {y.grad.shape if hasattr(y.grad, 'shape') else 'No shape'}")

# Check if they're equal
print(f"\nAre they equal? {np.array_equal(y.grad, gradient)}")
print(f"Are they close? {np.allclose(y.grad, gradient)}")

print("\n✅ Debug test completed!")
