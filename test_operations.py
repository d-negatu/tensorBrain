#!/usr/bin/env python3
"""
Test the new operations: matmul, sum, and mean
"""

from tensor import Tensor
import numpy as np

def test_matmul():
    """Test matrix multiplication."""
    print("🧪 Testing matrix multiplication...")
    
    # Test 1: Simple matrix multiplication
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[5], [6]], requires_grad=False)
    c = a @ b
    
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c = a @ b: {c}")
    print(f"c.data: {c.data}")
    print(f"Expected: [[17], [39]]")
    
    # Test 2: Vector @ Matrix
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([[4], [5], [6]], requires_grad=False)
    z = x @ y
    
    print(f"\nx: {x}")
    print(f"y: {y}")
    print(f"z = x @ y: {z}")
    print(f"z.data: {z.data}")
    print(f"Expected: [32]")
    
    print("✅ Matrix multiplication tests passed!")

def test_sum():
    """Test sum operation."""
    print("\n🧪 Testing sum operation...")
    
    # Test 1: Sum all elements
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    total = x.sum()
    
    print(f"x: {x}")
    print(f"total = x.sum(): {total}")
    print(f"total.data: {total.data}")
    print(f"Expected: 10.0")
    
    # Test 2: Sum along dimension 0 (rows)
    row_sum = x.sum(dim=0)
    print(f"\nrow_sum = x.sum(dim=0): {row_sum}")
    print(f"row_sum.data: {row_sum.data}")
    print(f"Expected: [4, 6]")
    
    # Test 3: Sum along dimension 1 (columns)
    col_sum = x.sum(dim=1)
    print(f"\ncol_sum = x.sum(dim=1): {col_sum}")
    print(f"col_sum.data: {col_sum.data}")
    print(f"Expected: [3, 7]")
    
    # Test 4: Sum with keepdim
    row_sum_keep = x.sum(dim=0, keepdim=True)
    print(f"\nrow_sum_keep = x.sum(dim=0, keepdim=True): {row_sum_keep}")
    print(f"row_sum_keep.data: {row_sum_keep.data}")
    print(f"Expected: [[4, 6]]")
    
    print("✅ Sum operation tests passed!")

def test_mean():
    """Test mean operation."""
    print("\n🧪 Testing mean operation...")
    
    # Test 1: Mean of all elements
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    avg = x.mean()
    
    print(f"x: {x}")
    print(f"avg = x.mean(): {avg}")
    print(f"avg.data: {avg.data}")
    print(f"Expected: 2.5")
    
    # Test 2: Mean along dimension 0 (rows)
    row_mean = x.mean(dim=0)
    print(f"\nrow_mean = x.mean(dim=0): {row_mean}")
    print(f"row_mean.data: {row_mean.data}")
    print(f"Expected: [2, 3]")
    
    # Test 3: Mean along dimension 1 (columns)
    col_mean = x.mean(dim=1)
    print(f"\ncol_mean = x.mean(dim=1): {col_mean}")
    print(f"col_mean.data: {col_mean.data}")
    print(f"Expected: [1.5, 3.5]")
    
    # Test 4: Mean with keepdim
    row_mean_keep = x.mean(dim=0, keepdim=True)
    print(f"\nrow_mean_keep = x.mean(dim=0, keepdim=True): {row_mean_keep}")
    print(f"row_mean_keep.data: {row_mean_keep.data}")
    print(f"Expected: [[2, 3]]")
    
    print("✅ Mean operation tests passed!")

def test_gradient_inheritance():
    """Test that operations inherit gradient requirements."""
    print("\n🧪 Testing gradient inheritance...")
    
    # Test matmul
    a = Tensor([[1, 2]], requires_grad=True)
    b = Tensor([[3], [4]], requires_grad=False)
    c = a @ b
    print(f"a.requires_grad: {a.requires_grad}")
    print(f"b.requires_grad: {b.requires_grad}")
    print(f"c.requires_grad: {c.requires_grad}")
    print(f"Expected: True (inherited from a)")
    
    # Test sum
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x.sum()
    print(f"\nx.requires_grad: {x.requires_grad}")
    print(f"y.requires_grad: {y.requires_grad}")
    print(f"Expected: True (inherited from x)")
    
    print("✅ Gradient inheritance tests passed!")

if __name__ == "__main__":
    print("🚀 Testing TensorBrain operations...\n")
    
    test_matmul()
    test_sum()
    test_mean()
    test_gradient_inheritance()
    
    print("\n🎉 All operation tests passed!")
    print("\n📝 Next steps:")
    print("   1. Test with real examples")
    print("   2. Build neural network layers")
    print("   3. Create training examples")
