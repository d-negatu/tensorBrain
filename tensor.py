import numpy as np
from typing import Optional, Union, List, Tuple


class Tensor:
    """
    Tensor class for TensorBrain - data structure for numerical computing with tensors.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, List, float, int],  # data is the actual numbers the tensor holds.
        requires_grad: bool = False,  # The loss function(err function) we are minimizing to the tensor's data.
        grad_fn: Optional['Function'] = None,  # The function that computes the gradient of the loss with respect to the tensor's data.
        device: str = 'cpu'  # The device on which the tensor is stored. 
    ):
        """
        Initialize a Tensor.

        Args:
            data: The data to store in the tensor.
            requires_grad: Whether to compute gradients for this tensor.
            grad_fn: The function that computes the gradient of the loss with respect to the tensor's data.
            device: The device on which the tensor is stored.
        """
        # Convert input to numpy array
        if isinstance(data, (int, float)):
            self.data = np.array([data], dtype=np.float32)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Autograd Properties
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
        self.device = device
        # For autograd - store references to input tensors
        self._saved_tensors = []
    
    def backward(self, gradient: Optional[np.ndarray] = None):
        """
        Compute gradients using backpropagation.
        
        Args:
            gradient: Gradient w.r.t. this tensor (defaults to ones)
        """
        # Step 1: Check if we need gradients
        if not self.requires_grad:
            return  # Skip if no gradients needed
            
        # Step 2: Handle default gradient
        if gradient is None:
            # Default gradient is ones with the same shape as the tensor
            gradient = np.ones_like(self.data)
        
        # Step 3: Initialize gradient storage
        if self.grad is None:
            self.grad = np.zeros_like(self.data)  # Create storage
        
        # Step 4: Accumulate gradients
        self.grad += gradient  # Add to existing gradients
        
        # Step 5: Backpropagate through computation graph
        if self.grad_fn is not None:
            self.grad_fn.backward(gradient)  # Continue chain
    
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad.fill(0)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.data.size
    
    def __repr__(self) -> str:
        """String representation of tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    

    def __add__(self, other):
        """
        Element-wise addition: self + other
        
        Args:
            other: Another tensor or scalar to add
            
        Returns:
            New Tensor with the sum
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if(self.data.shape != other.data.shape):
            raise ValueError(f"Shape mismatch: {self.data.shape} != {other.data.shape}")
        
        # Check if we need gradients
        requires_grad = self.requires_grad or other.requires_grad
        
        if requires_grad:
            # Create gradient function and use it
            grad_fn = AddFunction()
            result_data = grad_fn.forward(self, other)
            return Tensor(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
        else:
            # No gradients needed, just compute result
            result_data = self.data + other.data
            return Tensor(result_data, requires_grad=False)

    def __mul__(self, other):
        """Element-wise multiplication: self * other"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if(self.data.shape != other.data.shape):
            raise ValueError(f"Shape mismatch: {self.data.shape} != {other.data.shape}")
        
        # Check if we need gradients
        requires_grad = self.requires_grad or other.requires_grad
        
        if requires_grad:
            # Create gradient function and use it
            grad_fn = MulFunction()
            result_data = grad_fn.forward(self, other)
            return Tensor(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
        else:
            # No gradients needed, just compute result
            result_data = self.data * other.data
            return Tensor(result_data, requires_grad=False)

    def __matmul__(self, other):
        """Matrix multiplication: self @ other"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        # Check shape compatibility for matrix multiplication
        if self.data.shape[-1] != other.data.shape[0]:
            raise ValueError(f"Shapes {self.data.shape} and {other.data.shape} are not compatible for matrix multiplication")
        
        # Check if we need gradients
        requires_grad = self.requires_grad or other.requires_grad
        
        if requires_grad:
            # Create gradient function and use it
            grad_fn = MatMulFunction()
            result_data = grad_fn.forward(self, other)
            return Tensor(result_data, requires_grad=requires_grad, grad_fn=grad_fn)
        else:
            # No gradients needed, just compute result
            result_data = self.data @ other.data
            return Tensor(result_data, requires_grad=False)
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False):
        """
        Sum along dimension.
        
        Args:
            dim: Dimension to sum along (None = sum all elements)
            keepdim: Whether to keep the dimension (True) or remove it (False)
            
        Returns:
            New Tensor with summed values
        """
        if self.requires_grad:
            # Create gradient function and use it
            grad_fn = SumFunction()
            result_data = grad_fn.forward(self, dim, keepdim)
            return Tensor(result_data, requires_grad=self.requires_grad, grad_fn=grad_fn)
        else:
            # No gradients needed, just compute result
            if dim is None:
                result_data = np.sum(self.data)
            else:
                result_data = np.sum(self.data, axis=dim, keepdims=keepdim)
            return Tensor(result_data, requires_grad=False)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False):
        """
        Mean along dimension.
        
        Args:
            dim: Dimension to compute mean along (None = mean of all elements)
            keepdim: Whether to keep the dimension (True) or remove it (False)
            
        Returns:
            New Tensor with mean values
        """
        if self.requires_grad:
            # Create gradient function and use it
            grad_fn = MeanFunction()
            result_data = grad_fn.forward(self, dim, keepdim)
            return Tensor(result_data, requires_grad=self.requires_grad, grad_fn=grad_fn)
        else:
            # No gradients needed, just compute result
            if dim is None:
                result_data = np.mean(self.data)
            else:
                result_data = np.mean(self.data, axis=dim, keepdims=keepdim)
            return Tensor(result_data, requires_grad=False)


class Function:
    """Base class for autograd functions."""
    
    def __init__(self):
        self.saved_tensors = []
    
    def save_for_backward(self, *tensors):
        """Save tensors for backward pass."""
        self.saved_tensors = list(tensors)
    
    def forward(self, *args):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """Backward pass - to be implemented by subclasses."""
        raise NotImplementedError


class AddFunction(Function):
    """Gradient function for addition."""
    
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a.data + b.data
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        
        # Gradient of addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        grad_a = grad_output if a.requires_grad else None
        grad_b = grad_output if b.requires_grad else None
        
        if grad_a is not None:
            a.backward(grad_a)
        if grad_b is not None:
            b.backward(grad_b)


class MulFunction(Function):
    """Gradient function for multiplication."""
    
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a.data * b.data
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        
        # Gradient of multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        grad_a = grad_output * b.data if a.requires_grad else None
        grad_b = grad_output * a.data if b.requires_grad else None
        
        if grad_a is not None:
            a.backward(grad_a)
        if grad_b is not None:
            b.backward(grad_b)


class MatMulFunction(Function):
    """Gradient function for matrix multiplication."""
    
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a.data @ b.data
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        
        # Gradient of matrix multiplication: ∂(A@B)/∂A = grad_output @ B.T, ∂(A@B)/∂B = A.T @ grad_output
        grad_a = grad_output @ b.data.T if a.requires_grad else None
        grad_b = a.data.T @ grad_output if b.requires_grad else None
        
        if grad_a is not None:
            a.backward(grad_a)
        if grad_b is not None:
            b.backward(grad_b)


class SumFunction(Function):
    """Gradient function for sum operation."""
    
    def forward(self, input_tensor, dim=None, keepdim=False):
        self.save_for_backward(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
        return np.sum(input_tensor.data, axis=dim, keepdims=keepdim)
    
    def backward(self, grad_output):
        input_tensor, = self.saved_tensors
        
        if input_tensor.requires_grad:
            # Gradient of sum: ∂sum/∂x = 1 (broadcasted to input shape)
            if self.dim is None:
                # Sum over all elements
                grad_input = np.full_like(input_tensor.data, grad_output)
            else:
                # Sum over specific dimension
                grad_input = np.broadcast_to(grad_output, input_tensor.data.shape)
            
            input_tensor.backward(grad_input)


class MeanFunction(Function):
    """Gradient function for mean operation."""
    
    def forward(self, input_tensor, dim=None, keepdim=False):
        self.save_for_backward(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
        return np.mean(input_tensor.data, axis=dim, keepdims=keepdim)
    
    def backward(self, grad_output):
        input_tensor, = self.saved_tensors
        
        if input_tensor.requires_grad:
            # Gradient of mean: ∂mean/∂x = 1/n (broadcasted to input shape)
            if self.dim is None:
                # Mean over all elements
                n = input_tensor.data.size
                grad_input = np.full_like(input_tensor.data, grad_output / n)
            else:
                # Mean over specific dimension
                n = input_tensor.data.shape[self.dim]
                grad_input = np.broadcast_to(grad_output / n, input_tensor.data.shape)
            
            input_tensor.backward(grad_input)
