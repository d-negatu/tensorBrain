# Tensor Guide

## Overview

This guide provides comprehensive documentation on working with tensors in TensorBrain.

## Creating Tensors

### From Python Lists

```python
import tensorbrain as tb

# Create from list
t = tb.tensor([1, 2, 3])

# Create 2D tensor
t = tb.tensor([[1, 2], [3, 4]])
```

### Using Special Functions

```python
# Zeros and ones
t = tb.zeros((3, 4))
t = tb.ones((2, 3, 4))

# Random tensors
t = tb.randn((3, 4))  # Normal distribution
t = tb.rand((3, 4))   # Uniform distribution
```

## Tensor Operations

### Arithmetic Operations

```python
a = tb.tensor([1, 2, 3])
b = tb.tensor([4, 5, 6])

# Element-wise operations
c = a + b
c = a - b
c = a * b
c = a / b
```

### Matrix Operations

```python
A = tb.tensor([[1, 2], [3, 4]])
B = tb.tensor([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B
C = tb.matmul(A, B)
```

### Reduction Operations

```python
t = tb.tensor([[1, 2, 3], [4, 5, 6]])

# Sum all elements
sum_all = t.sum()

# Sum along axis
sum_axis0 = t.sum(dim=0)
sum_axis1 = t.sum(dim=1)

# Other reductions
mean = t.mean()
max_val = t.max()
min_val = t.min()
```

### Reshaping

```python
t = tb.tensor([[1, 2, 3], [4, 5, 6]])

# Reshape
t_reshaped = t.reshape((3, 2))

# Flatten
t_flat = t.flatten()

# View (same data, different shape)
t_view = t.view((3, 2))
```

## Indexing and Slicing

```python
t = tb.tensor([[1, 2, 3], [4, 5, 6]])

# Basic indexing
elem = t[0, 1]  # Element at row 0, col 1

# Slicing
row = t[0, :]   # First row
col = t[:, 0]   # First column
subset = t[0:1, 1:3]  # Subset
```

## Data Types

```python
# Specify dtype
t = tb.tensor([1, 2, 3], dtype='float32')
t = tb.zeros((3, 4), dtype='float64')

# Convert dtype
t_float = t.float()
t_int = t.int()
```

## Broadcasting

```python
a = tb.tensor([[1, 2, 3]])      # Shape: (1, 3)
b = tb.tensor([[1], [2], [3]])   # Shape: (3, 1)

# Broadcasts to (3, 3)
c = a + b
```

## Device Placement

```python
# CPU tensor
t_cpu = tb.tensor([1, 2, 3])

# GPU tensor (if available)
t_gpu = tb.tensor([1, 2, 3]).to('cuda')

# Move between devices
t_cpu = t_gpu.to('cpu')
```

## Advanced Topics

### Automatic Differentiation

```python
t = tb.tensor([1.0, 2.0, 3.0], requires_grad=True)
result = (t ** 2).sum()
result.backward()
print(t.grad)
```

### Tensor Slicing and Indexing

See the [API Reference](./API_REFERENCE.md) for more detailed information on indexing operations.

## Best Practices

1. **Memory Management**: Use `.contiguous()` for non-contiguous tensors before operations
2. **Data Types**: Choose appropriate data types for your use case
3. **Device Consistency**: Ensure all tensors in an operation are on the same device
4. **Broadcasting**: Understand broadcasting rules to avoid unexpected behavior

## Troubleshooting

Common issues and solutions can be found in the [FAQ](../README.md#faq).

## See Also

- [API Reference](./API_REFERENCE.md)
- [Neural Network Modules](./NN_GUIDE.md)
- [Contributing Guide](../CONTRIBUTING.md)
