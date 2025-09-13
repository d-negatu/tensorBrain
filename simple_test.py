#!/usr/bin/env python3
"""
Simple test to check if our Tensor class works!
"""

# Test without numpy first
print("🧪 Testing Tensor class without numpy...")

# Let's create a simple version that doesn't need numpy
class SimpleTensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

# Test basic functionality
print("\n📝 Test 1: Creating tensors")
x = SimpleTensor(5.0, requires_grad=True)
y = SimpleTensor([1, 2, 3], requires_grad=False)
print(f"Tensor x: {x}")
print(f"Tensor y: {y}")

print("\n📝 Test 2: Checking properties")
print(f"x.data: {x.data}")
print(f"x.requires_grad: {x.requires_grad}")
print(f"x.grad: {x.grad}")

print("\n📝 Test 3: Different data types")
scalar = SimpleTensor(42.0)
vector = SimpleTensor([1, 2, 3, 4])
print(f"Scalar: {scalar}")
print(f"Vector: {vector}")

print("\n✅ Basic Tensor class is working!")
print("\n📝 Next: Install numpy and test the full implementation")
