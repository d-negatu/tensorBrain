#!/usr/bin/env python3
"""
Test basic Tensor functionality.
Run this to verify our Tensor class works correctly.
"""

from tensor import Tensor
import numpy as np

def test_tensor_creation():
    """Test creating tensors from different data types."""
    print("🧪 Testing tensor creation...")
    
    # Test scalar
    t1 = Tensor(5.0)
    print(f"Scalar tensor: {t1}")
    assert t1.data == 5.0
    assert t1.shape == (1,)
    
    # Test list
    t2 = Tensor([1, 2, 3, 4])
    print(f"List tensor: {t2}")
    assert np.array_equal(t2.data, [1, 2, 3, 4])
    assert t2.shape == (4,)
    
    # Test numpy array
    t3 = Tensor(np.array([[1, 2], [3, 4]]))
    print(f"2D tensor: {t3}")
    assert t3.shape == (2, 2)
    
    # Test requires_grad
    t4 = Tensor([1, 2, 3], requires_grad=True)
    print(f"Tensor with grad: {t4}")
    assert t4.requires_grad == True
    assert t4.grad is None  # Should be None initially
    
    print("✅ Tensor creation tests passed!")

def test_gradient_initialization():
    """Test gradient initialization and zero_grad."""
    print("\n🧪 Testing gradient initialization...")
    
    t = Tensor([1, 2, 3], requires_grad=True)
    
    # Initially no gradient
    assert t.grad is None
    
    # After backward, gradient should be initialized
    t.backward()
    assert t.grad is not None
    assert np.array_equal(t.grad, [1, 1, 1])  # Default gradient is ones
    
    # Test zero_grad
    t.zero_grad()
    assert np.array_equal(t.grad, [0, 0, 0])
    
    print("✅ Gradient initialization tests passed!")

def test_properties():
    """Test tensor properties."""
    print("\n🧪 Testing tensor properties...")
    
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.size == 6
    
    print("✅ Property tests passed!")

if __name__ == "__main__":
    print("🚀 Running TensorBrain basic tests...\n")
    
    test_tensor_creation()
    test_gradient_initialization()
    test_properties()
    
    print("\n🎉 All basic tests passed! Tensor class is working correctly.")
    print("\n📝 Next steps:")
    print("   1. Implement basic operations (add, mul, matmul)")
    print("   2. Build the autograd engine")
    print("   3. Test with simple computations")
