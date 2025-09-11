import numpy as np
from typing import Optional, Union, List, Tuple

def __init__(
    """
    Tensor class for TensorBrain - data structure for numerical computing with tensors.
    """
    self,
    data: Union[np.ndarray, List, float, int] # data is the actual numbers the tensor holds.
    requires_grad: bool = False, # The loss function(err function) we are minimizing to the tensor's data.
    grad_fn : Optional['Function'] = None, # The function that computes the gradient of the loss with respect to the tensor's data.
    device: str = 'cpu' # The device on which the tensor is stored. 
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
        self.data = np.array([data],dtype=np.float32)
    elif(isinstance(data, list)):
        self.data = np.array(data,dtype = np.float32)
    elif(isinstance(data, np.ndarray)):
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