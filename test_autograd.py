#!/usr/bin/env python3
"""
Test script for TensorBrain autograd engine.
"""

from tensor import Tensor
import numpy as np

def test_simple_addition():
    """Test basic addition with gradients."""
    print("Testing simple addition...")
    
    # Create tensors that require gradients
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([1.0, 4.0], requires_grad=True)
    
    # Perform addition
    c = a + b
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = a + b = {c}")
    
    # Compute gradients
    c.backward()
    
    print(f"∂c/∂a = {a.grad}")
    print(f"∂c/∂b = {b.grad}")
    
    # Expected: ∂c/∂a = [1, 1], ∂c/∂b = [1, 1]
    expected_grad = np.array([1.0, 1.0])
    assert np.allclose(a.grad, expected_grad), f"Expected {expected_grad}, got {a.grad}"
    assert np.allclose(b.grad, expected_grad), f"Expected {expected_grad}, got {b.grad}"
    print("✅ Addition test passed!")

def test_multiplication():
    """Test multiplication with gradients."""
    print("\nTesting multiplication...")
    
    # Create tensors that require gradients
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([1.0, 4.0], requires_grad=True)
    
    # Perform multiplication
    c = a * b
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = a * b = {c}")
    
    # Compute gradients
    c.backward()
    
    print(f"∂c/∂a = {a.grad}")
    print(f"∂c/∂b = {b.grad}")
    
    # Expected: ∂c/∂a = b = [1, 4], ∂c/∂b = a = [2, 3]
    expected_grad_a = np.array([1.0, 4.0])
    expected_grad_b = np.array([2.0, 3.0])
    assert np.allclose(a.grad, expected_grad_a), f"Expected {expected_grad_a}, got {a.grad}"
    assert np.allclose(b.grad, expected_grad_b), f"Expected {expected_grad_b}, got {b.grad}"
    print("✅ Multiplication test passed!")

def test_matrix_multiplication():
    """Test matrix multiplication with gradients."""
    print("\nTesting matrix multiplication...")
    
    # Create matrices that require gradients
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[2.0, 0.0], [1.0, 3.0]], requires_grad=True)
    
    # Perform matrix multiplication
    C = A @ B
    
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = A @ B = {C}")
    
    # Compute gradients
    C.backward()
    
    print(f"∂C/∂A = {A.grad}")
    print(f"∂C/∂B = {B.grad}")
    
    # Expected: ∂C/∂A = grad_output @ B.T, ∂C/∂B = A.T @ grad_output
    # Since grad_output is ones matrix, ∂C/∂A = B.T, ∂C/∂B = A.T
    expected_grad_A = np.array([[2.0, 1.0], [2.0, 1.0]])
    expected_grad_B = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(A.grad, expected_grad_A), f"Expected {expected_grad_A}, got {A.grad}"
    assert np.allclose(B.grad, expected_grad_B), f"Expected {expected_grad_B}, got {B.grad}"
    print("✅ Matrix multiplication test passed!")

def test_sum():
    """Test sum operation with gradients."""
    print("\nTesting sum operation...")
    
    # Create tensor that requires gradients
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Perform sum
    y = x.sum()
    
    print(f"x = {x}")
    print(f"y = sum(x) = {y}")
    
    # Compute gradients
    y.backward()
    
    print(f"∂y/∂x = {x.grad}")
    
    # Expected: ∂sum/∂x = 1 (broadcasted to input shape)
    expected_grad = np.ones_like(x.data)
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✅ Sum test passed!")

def test_chain_rule():
    """Test chain rule with multiple operations."""
    print("\nTesting chain rule...")
    
    # Create tensor that requires gradients
    x = Tensor([2.0], requires_grad=True)
    
    # Chain: x -> x^2 -> x^2 + 1 -> sum
    y = x * x  # x^2
    z = y + Tensor([1.0])  # x^2 + 1
    w = z.sum()  # sum(x^2 + 1)
    
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    print(f"z = x^2 + 1 = {z}")
    print(f"w = sum(x^2 + 1) = {w}")
    
    # Compute gradients
    w.backward()
    
    print(f"∂w/∂x = {x.grad}")
    
    # Expected: ∂w/∂x = ∂w/∂z * ∂z/∂y * ∂y/∂x = 1 * 1 * 2x = 2x = 4
    expected_grad = np.array([4.0])
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print("✅ Chain rule test passed!")

if __name__ == "__main__":
    print("🚀 Testing TensorBrain Autograd Engine")
    print("=" * 50)
    
    try:
        test_simple_addition()
        test_multiplication()
        test_matrix_multiplication()
        test_sum()
        test_chain_rule()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Autograd engine is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
