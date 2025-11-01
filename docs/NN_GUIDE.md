# Neural Network Modules Guide

## Overview

Comprehensive guide to building neural networks with TensorBrain modules.

## Basic Layers

### Linear Layer

```python
import tensorbrain as tb

# Create a linear layer
layer = tb.nn.Linear(in_features=10, out_features=5)

# Forward pass
output = layer(input_tensor)  # input shape: (batch_size, 10)
```

### Convolutional Layers

#### Conv1d (1D Convolution)

```python
conv1d = tb.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv1d(input_tensor)  # input shape: (batch_size, 3, seq_len)
```

#### Conv2d (2D Convolution)

```python
conv2d = tb.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv2d(input_tensor)  # input shape: (batch_size, 3, height, width)
```

#### Conv3d (3D Convolution)

```python
conv3d = tb.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv3d(input_tensor)  # input shape: (batch_size, 3, depth, height, width)
```

### Pooling Layers

```python
# Max pooling
max_pool = tb.nn.MaxPool2d(kernel_size=2, stride=2)
output = max_pool(input_tensor)

# Average pooling
avg_pool = tb.nn.AvgPool2d(kernel_size=2, stride=2)
output = avg_pool(input_tensor)
```

## Activation Functions

```python
# ReLU
relu = tb.nn.ReLU()
output = relu(x)

# GELU (Gaussian Error Linear Unit)
gelu = tb.nn.GELU()
output = gelu(x)

# Sigmoid
sigmoid = tb.nn.Sigmoid()
output = sigmoid(x)

# Tanh
tanh = tb.nn.Tanh()
output = tanh(x)

# Softmax
softmax = tb.nn.Softmax(dim=1)
output = softmax(x)
```

## Normalization Layers

```python
# Batch Normalization
batch_norm = tb.nn.BatchNorm1d(num_features=64)
output = batch_norm(x)

# Layer Normalization
layer_norm = tb.nn.LayerNorm(normalized_shape=64)
output = layer_norm(x)
```

## Dropout

```python
# Dropout layer
dropout = tb.nn.Dropout(p=0.5)
output = dropout(x)  # During training
output = dropout(x, training=False)  # During inference
```

## Building a Complete Network

```python
class SimpleCNN(tb.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = tb.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = tb.nn.ReLU()
        self.pool1 = tb.nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = tb.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = tb.nn.ReLU()
        self.pool2 = tb.nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = tb.nn.Linear(64 * 8 * 8, 128)
        self.relu3 = tb.nn.ReLU()
        self.dropout = tb.nn.Dropout(p=0.5)
        self.fc2 = tb.nn.Linear(128, 10)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten and fully connected layers
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Usage
model = SimpleCNN()
output = model(input_tensor)
```

## Transfer Learning

```python
# Load pretrained model
model = tb.models.resnet50(pretrained=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = tb.nn.Linear(2048, num_classes)

# Only final layer is trainable
```

## Recurrent Neural Networks

```python
# LSTM
lstm = tb.nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(input_tensor)

# GRU
gru = tb.nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, h_n = gru(input_tensor)
```

## Transformer Modules

```python
# Multi-head attention
attention = tb.nn.MultiheadAttention(embed_dim=512, num_heads=8)
output, attn_weights = attention(query, key, value)

# Transformer encoder layer
encoder_layer = tb.nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
output = encoder_layer(x)
```

## Best Practices

1. **Weight Initialization**: Use appropriate initialization strategies for different layer types
2. **Batch Normalization**: Apply after convolutions, before activation
3. **Dropout**: Use during training for regularization
4. **Layer Organization**: Group related layers using Sequential or ModuleList

## See Also

- [API Reference](./API_REFERENCE.md)
- [Tensor Guide](./TENSOR_GUIDE.md)
- [Contributing Guide](../CONTRIBUTING.md)
