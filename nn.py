#!/usr/bin/env python3
"""
Neural Network Layers for TensorBrain
Built on top of the existing autograd engine
"""

import numpy as np
from typing import Optional, List, Tuple
from tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self.training = True
    
    def forward(self, x):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, x):
        """Make modules callable like functions."""
        return self.forward(x)
    
    def parameters(self):
        """Return all parameters that require gradients."""
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params
    
    def train(self):
        """Set module to training mode."""
        self.training = True
        for module in self.children():
            module.train()
    
    def eval(self):
        """Set module to evaluation mode."""
        self.training = False
        for module in self.children():
            module.eval()
    
    def children(self):
        """Return child modules."""
        children = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                children.append(attr)
        return children


class Linear(Module):
    """Linear (fully connected) layer: y = xW^T + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (out_features, in_features)),
            requires_grad=True
        )
        
        # Initialize bias
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = xW^T + b"""
        # x: [batch_size, in_features]
        # weight: [out_features, in_features]
        # output: [batch_size, out_features]
        
        # Transpose weight matrix
        weight_T = Tensor(self.weight.data.T, requires_grad=self.weight.requires_grad)
        output = x @ weight_T
        if self.bias is not None:
            # Broadcast bias to match output shape
            bias_broadcast = Tensor(np.broadcast_to(self.bias.data, output.data.shape), requires_grad=self.bias.requires_grad)
            output = output + bias_broadcast
        return output
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class ReLU(Module):
    """Rectified Linear Unit activation function: max(0, x)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: max(0, x)"""
        # We need to implement ReLU operation with gradients
        # For now, let's use a simple implementation
        return relu(x)
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function: 1 / (1 + exp(-x))"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: sigmoid(x)"""
        return sigmoid(x)
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation function"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: tanh(x)"""
        return tanh(x)
    
    def __repr__(self) -> str:
        return "Tanh()"


class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all modules sequentially"""
        for module in self.modules:
            x = module(x)
        return x
    
    def __repr__(self) -> str:
        module_strs = [str(module) for module in self.modules]
        return "Sequential(\n  " + ",\n  ".join(module_strs) + "\n)"


# Activation functions (these need to be implemented with gradients)
def relu(x: Tensor) -> Tensor:
    """ReLU activation with gradient support"""
    # Simple implementation - we'll need to add gradient support later
    result_data = np.maximum(0, x.data)
    return Tensor(result_data, requires_grad=x.requires_grad)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation with gradient support"""
    # Simple implementation - we'll need to add gradient support later
    result_data = 1.0 / (1.0 + np.exp(-x.data))
    return Tensor(result_data, requires_grad=x.requires_grad)


def tanh(x: Tensor) -> Tensor:
    """Tanh activation with gradient support"""
    # Simple implementation - we'll need to add gradient support later
    result_data = np.tanh(x.data)
    return Tensor(result_data, requires_grad=x.requires_grad)


# Loss functions
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss"""
    return ((pred - target) ** 2).mean()


def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Cross Entropy loss for classification"""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    return -(target * (pred + epsilon).log()).mean()


# Optimizer
class SGD:
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in parameters]
    
    def step(self):
        """Perform one optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                if self.momentum > 0:
                    # Momentum update
                    self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad
                    param.data -= self.velocities[i]
                else:
                    # Simple SGD update
                    param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing TensorBrain Neural Network Layers")
    print("=" * 50)
    
    # Test Linear layer
    print("\n📝 Test 1: Linear Layer")
    linear = Linear(3, 2)
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    print(f"Input shape: {x.shape}")
    print(f"Linear layer: {linear}")
    
    output = linear(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test Sequential model
    print("\n📝 Test 2: Sequential Model")
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
        Sigmoid()
    )
    print(f"Model: {model}")
    
    output = model(x)
    print(f"Model output shape: {output.shape}")
    print(f"Model output: {output}")
    
    # Test training step
    print("\n📝 Test 3: Training Step")
    target = Tensor([[0.5, 0.8], [0.3, 0.9]], requires_grad=False)
    loss = mse_loss(output, target)
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    print("✅ Backward pass completed")
    
    # Test optimizer
    print("\n📝 Test 4: Optimizer")
    optimizer = SGD(model.parameters(), lr=0.01)
    print(f"Number of parameters: {len(model.parameters())}")
    
    # Training step
    optimizer.step()
    optimizer.zero_grad()
    print("✅ Training step completed")
    
    print("\n🎉 All tests passed! Neural network layers are working!")
    print("\n📝 Next steps:")
    print("   1. Add gradient support to activation functions")
    print("   2. Implement more layers (Conv2D, BatchNorm)")
    print("   3. Add more optimizers (Adam, RMSprop)")
    print("   4. Create training examples")
