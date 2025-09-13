#!/usr/bin/env python3
"""
Test the backward method to make sure it works correctly!
"""

from tensor import Tensor
import numpy as np

def test_backward_basic():
    """Test basic backward functionality."""
    print("🧪 Testing basic backward functionality...")
    
    # Test 1: Scalar tensor with requires_grad=True
    x = Tensor(5.0, requires_grad=True)
    print(f"Created tensor: {x}")
    print(f"Initial grad: {x.grad}")
    
    # Call backward
    x.backward()
    print(f"After backward: grad = {x.grad}")
    print(f"Expected: [1.0]")
    assert np.array_equal(x.grad, [1.0]), f"Expected [1.0], got {x.grad}"
    
    # Test 2: Vector tensor with requires_grad=True
    y = Tensor([1, 2, 3], requires_grad=True)
    print(f"\nCreated tensor: {y}")
    print(f"Initial grad: {y.grad}")
    
    # Call backward with gradient
    gradient = np.array([0.1, 0.2, 0.3])
    y.backward(gradient)
    print(f"After backward with gradient {gradient}: grad = {y.grad}")
    print(f"Expected: [0.1, 0.2, 0.3]")
    assert np.array_equal(y.grad, gradient), f"Expected {gradient}, got {y.grad}"
    
    # Test 3: Tensor with requires_grad=False
    z = Tensor([4, 5, 6], requires_grad=False)
    print(f"\nCreated tensor: {z}")
    print(f"Initial grad: {z.grad}")
    
    # Call backward (should do nothing)
    z.backward([0.1, 0.2, 0.3])
    print(f"After backward: grad = {z.grad}")
    print(f"Expected: None (no gradients computed)")
    assert z.grad is None, f"Expected None, got {z.grad}"
    
    print("\n✅ Basic backward tests passed!")

def test_gradient_accumulation():
    """Test gradient accumulation."""
    print("\n🧪 Testing gradient accumulation...")
    
    x = Tensor(2.0, requires_grad=True)
    print(f"Created tensor: {x}")
    
    # First backward call
    x.backward()
    print(f"After first backward: grad = {x.grad}")
    
    # Second backward call (should accumulate)
    x.backward()
    print(f"After second backward: grad = {x.grad}")
    print(f"Expected: [2.0] (1.0 + 1.0)")
    assert np.array_equal(x.grad, [2.0]), f"Expected [2.0], got {x.grad}"
    
    print("✅ Gradient accumulation test passed!")

def test_zero_grad():
    """Test zero_grad functionality."""
    print("\n🧪 Testing zero_grad functionality...")
    
    x = Tensor(3.0, requires_grad=True)
    x.backward()
    print(f"After backward: grad = {x.grad}")
    
    x.zero_grad()
    print(f"After zero_grad: grad = {x.grad}")
    print(f"Expected: [0.0]")
    assert np.array_equal(x.grad, [0.0]), f"Expected [0.0], got {x.grad}"
    
    print("✅ Zero grad test passed!")

def test_error_handling():
    """Test error handling."""
    print("\n🧪 Testing error handling...")
    
    # Test non-scalar tensor without gradient
    x = Tensor([1, 2, 3], requires_grad=True)
    try:
        x.backward()  # Should raise RuntimeError
        assert False, "Expected RuntimeError for non-scalar tensor without gradient"
    except RuntimeError as e:
        print(f"✅ Correctly caught error: {e}")
    
    print("✅ Error handling test passed!")

if __name__ == "__main__":
    print("🚀 Testing TensorBrain backward method...\n")
    
    test_backward_basic()
    test_gradient_accumulation()
    test_zero_grad()
    test_error_handling()
    
    print("\n🎉 All backward method tests passed!")
    print("\n📝 Next steps:")
    print("   1. Implement the Function class")
    print("   2. Add basic operations (add, mul, matmul)")
    print("   3. Test with simple computations")
