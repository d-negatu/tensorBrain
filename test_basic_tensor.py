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
    
    # Test scalar tensor (should work with default gradient)
    t_scalar = Tensor(5.0, requires_grad=True)
    
    # Initially no gradient
    assert t_scalar.grad is None
    
    # After backward, gradient should be initialized
    t_scalar.backward()
    assert t_scalar.grad is not None
    assert np.array_equal(t_scalar.grad, [1.0])  # Default gradient is 1 for scalars
    
    # Test zero_grad
    t_scalar.zero_grad()
    assert np.array_equal(t_scalar.grad, [0.0])
    
    # Test non-scalar tensor (should require explicit gradient)
    t_vector = Tensor([1, 2, 3], requires_grad=True)
    assert t_vector.grad is None
    
    # This should work with explicit gradient
    gradient = np.array([0.1, 0.2, 0.3])
    t_vector.backward(gradient)
    assert t_vector.grad is not None
    # Check if gradients are close (allowing for small floating point differences)
    assert np.allclose(t_vector.grad, gradient), f"Expected {gradient}, got {t_vector.grad}"
    
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
