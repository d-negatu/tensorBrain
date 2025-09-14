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
            # For scalar tensors, gradient is 1
            if self.data.size == 1:
                gradient = np.ones_like(self.data)  # Set to 1
            else:
                raise RuntimeError("gradient must be specified for non-scalar tensors")
        
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
    
    def __matmul__(self, other):
        """Matrix multiplication: self @ other"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        # Check shape compatibility for matrix multiplication
        if self.data.shape[-1] != other.data.shape[0]:
            raise ValueError(f"Shapes {self.data.shape} and {other.data.shape} are not compatible for matrix multiplication")
        
        result_data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, requires_grad=requires_grad)
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False):
        """
        Sum along dimension.
        
        Args:
            dim: Dimension to sum along (None = sum all elements)
            keepdim: Whether to keep the dimension (True) or remove it (False)
            
        Returns:
            New Tensor with summed values
        """
        if dim is None:
            # Sum all elements into a scalar
            result_data = np.sum(self.data)
            return Tensor(result_data, requires_grad=self.requires_grad)
        else:
            # Sum along specified dimension
            result_data = np.sum(self.data, axis=dim, keepdims=keepdim)
            return Tensor(result_data, requires_grad=self.requires_grad)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False):
        """
        Mean along dimension.
        
        Args:
            dim: Dimension to compute mean along (None = mean of all elements)
            keepdim: Whether to keep the dimension (True) or remove it (False)
            
        Returns:
            New Tensor with mean values
        """
        if dim is None:
            # Mean of all elements into a scalar
            result_data = np.mean(self.data)
            return Tensor(result_data, requires_grad=self.requires_grad)
        else:
            # Mean along specified dimension
            result_data = np.mean(self.data, axis=dim, keepdims=keepdim)
            return Tensor(result_data, requires_grad=self.requires_grad)

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
        result_data = self.data + other.data
        # Gradient inheritance over neural networks
        requires_grad = self.requires_grad or other.requires_grad

        return Tensor(result_data, requires_grad=requires_grad)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if(self.data.shape != other.data.shape):
            raise ValueError(f"Shape mismatch: {self.data.shape} != {other.data.shape}")
        result_data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, requires_grad=requires_grad)

    def __matmul__(self, other):
        """Matrix multiplication: self @ other"""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        # Check shape compatibility for matrix multiplication
        if self.data.shape[-1] != other.data.shape[0]:
            raise ValueError(f"Shapes {self.data.shape} and {other.data.shape} are not compatible for matrix multiplication")
        
        result_data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, requires_grad=requires_grad)
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False):
        """
        Sum along dimension.
    
    Args:
        dim: Dimension to sum along (None = sum all elements)
        keepdim: Whether to keep the dimension (True) or remove it (False)
        
    Returns:
        New Tensor with summed values
        """
        if dim is None:
            # Sum all elements into a scalar
            result_data = np.sum(self.data)
            # Always returns a scalar tensor
            return Tensor(result_data, requires_grad=self.requires_grad)
        else:
            # Sum along specified dimension
            result_data = np.sum(self.data, axis=dim, keepdims=keepdim)
            return Tensor(result_data, requires_grad=self.requires_grad)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False):
        """
        Mean along dimension.
        
        Args:
            dim: Dimension to compute mean along (None = mean of all elements)
            keepdim: Whether to keep the dimension (True) or remove it (False)
            
        Returns:
            New Tensor with mean values
        """
        if dim is None:
            # Mean of all elements into a scalar
            result_data = np.mean(self.data)
            return Tensor(result_data, requires_grad=self.requires_grad)
        else:
            # Mean along specified dimension
            result_data = np.mean(self.data, axis=dim, keepdims=keepdim)
            return Tensor(result_data, requires_grad=self.requires_grad)


class Function:
    """Base class for autograd functions."""
    pass
