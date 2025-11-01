# TensorBrain API Reference

## Overview

Comprehensive API documentation for TensorBrain library.

## Table of Contents

1. [Tensor Operations](#tensor-operations)
2. [Neural Network Modules](#neural-network-modules)
3. [Loss Functions](#loss-functions)
4. [Optimizers](#optimizers)
5. [Utilities](#utilities)

## Tensor Operations

### Basic Operations

```python
# Creation
tensor = tb.tensor([[1, 2], [3, 4]])
zeros = tb.zeros((3, 4))
ones = tb.ones((3, 4))

# Operations
result = tensor + other
result = tensor @ other
result = tensor.sum()
```

### Advanced Operations

- Matrix operations
- Reduction operations
- Indexing and slicing
- Broadcasting

## Neural Network Modules

### Linear Layer

```python
linear = tb.nn.Linear(in_features=10, out_features=5)
output = linear(input)
```

### Convolutional Layers

```python
conv = tb.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
output = conv(input)
```

### Activation Functions

- ReLU
- GELU
- Sigmoid
- Tanh
- Softmax

## Loss Functions

- CrossEntropyLoss
- MSELoss
- L1Loss
- BCELoss

## Optimizers

- SGD
- Adam
- AdamW
- RMSprop

## Utilities

- Model checkpointing
- Training loops
- Evaluation utilities

## Contributing

For more detailed information, see the [Contributing Guide](../CONTRIBUTING.md).
