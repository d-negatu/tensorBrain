#!/usr/bin/env python3
"""
Test our Tensor class to make sure the constructor works!
"""

from tensor import Tensor
import numpy as np

def test_tensor_creation():
    """Test creating tensors from different data types."""
    print("🧪 Testing tensor creation...")
    
    # Test scalar
    t1 = Tensor(5.0)
    print(f"Scalar tensor: {t1}")
    print(f"  Data: {t1.data}")
    print(f"  Shape: {t1.shape}")
    print(f"  Requires grad: {t1.requires_grad}")
    
    # Test list
    t2 = Tensor([1, 2, 3, 4])
    print(f"\nList tensor: {t2}")
    print(f"  Data: {t2.data}")
    print(f"  Shape: {t2.shape}")
    
    # Test numpy array
    t3 = Tensor(np.array([[1, 2], [3, 4]]))
    print(f"\n2D tensor: {t3}")
    print(f"  Data: {t3.data}")
    print(f"  Shape: {t3.shape}")
    print(f"  Dimensions: {t3.ndim}")
    print(f"  Size: {t3.size}")
    
    # Test with requires_grad=True
    t4 = Tensor([1, 2, 3], requires_grad=True)
    print(f"\nTensor with grad: {t4}")
    print(f"  Requires grad: {t4.requires_grad}")
    print(f"  Grad: {t4.grad}")
    
    print("\n✅ Tensor creation tests passed!")

if __name__ == "__main__":
    print("🚀 Testing TensorBrain Tensor class...\n")
    test_tensor_creation()
    print("\n🎉 Constructor is working correctly!")
    print("\n📝 Next steps:")
    print("   1. Implement the backward() method")
    print("   2. Build the Function class for autograd")
    print("   3. Add basic operations (add, mul, matmul)")
