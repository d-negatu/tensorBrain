#!/usr/bin/env python3
"""
Computer Vision Layers for TensorBrain
Conv2D, Pooling, and other CV layers
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss


class Conv2D(Module):
    """2D Convolutional layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights using Xavier initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Conv2D layer"""
        # x: [batch, channels, height, width]
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Add padding if needed
        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), 
                                     (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x.data
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input window
                        ih_start = oh * self.stride
                        iw_start = ow * self.stride
                        ih_end = ih_start + self.kernel_size
                        iw_end = iw_start + self.kernel_size
                        
                        # Extract input window
                        input_window = x_padded[b, :, ih_start:ih_end, iw_start:iw_end]
                        
                        # Convolve with kernel
                        output[b, oc, oh, ow] = np.sum(input_window * self.weight.data[oc])
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.data.reshape(1, -1, 1, 1)
        
        return Tensor(output, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class MaxPool2D(Module):
    """2D Max Pooling layer"""
    
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MaxPool2D layer"""
        # x: [batch, channels, height, width]
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Add padding if needed
        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), 
                                     (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x.data
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input window
                        ih_start = oh * self.stride
                        iw_start = ow * self.stride
                        ih_end = ih_start + self.kernel_size
                        iw_end = iw_start + self.kernel_size
                        
                        # Extract input window and take max
                        input_window = x_padded[b, c, ih_start:ih_end, iw_start:iw_end]
                        output[b, c, oh, ow] = np.max(input_window)
        
        return Tensor(output, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2D(Module):
    """2D Average Pooling layer"""
    
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through AvgPool2D layer"""
        # x: [batch, channels, height, width]
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Add padding if needed
        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), 
                                     (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x.data
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input window
                        ih_start = oh * self.stride
                        iw_start = ow * self.stride
                        ih_end = ih_start + self.kernel_size
                        iw_end = iw_start + self.kernel_size
                        
                        # Extract input window and take average
                        input_window = x_padded[b, c, ih_start:ih_end, iw_start:iw_end]
                        output[b, c, oh, ow] = np.mean(input_window)
        
        return Tensor(output, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return f"AvgPool2D(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class Flatten(Module):
    """Flatten layer for CNN to FC transition"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Flatten layer"""
        # x: [batch, channels, height, width] -> [batch, channels * height * width]
        batch_size = x.shape[0]
        flattened = x.data.reshape(batch_size, -1)
        return Tensor(flattened, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return "Flatten()"


def create_cnn_model(input_channels: int = 3, num_classes: int = 10) -> Sequential:
    """Create a simple CNN model"""
    return Sequential(
        Conv2D(input_channels, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        
        Conv2D(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        
        Conv2D(64, 128, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        
        Flatten(),
        Linear(128 * 4 * 4, 512),  # Assuming 32x32 input -> 4x4 after pooling
        ReLU(),
        Linear(512, num_classes)
    )


def create_sample_image_data(batch_size: int = 8, channels: int = 3, 
                           height: int = 32, width: int = 32, num_classes: int = 10) -> List[Tuple[Tensor, Tensor]]:
    """Create sample image data for testing"""
    data = []
    
    for _ in range(batch_size):
        # Create random image
        image = Tensor(np.random.randn(channels, height, width), requires_grad=False)
        
        # Create random label
        label = Tensor(np.random.randint(0, num_classes, (1,)), requires_grad=False)
        
        data.append((image, label))
    
    return data


def benchmark_cnn(model: Sequential, data_loader: List[Tuple[Tensor, Tensor]]) -> Dict[str, float]:
    """Benchmark CNN performance"""
    print("📊 Benchmarking CNN...")
    
    # Forward pass timing
    start_time = time.time()
    for image, _ in data_loader:
        # Add batch dimension
        image_batch = Tensor(image.data.reshape(1, *image.shape), requires_grad=False)
        output = model(image_batch)
    forward_time = (time.time() - start_time) / len(data_loader)
    
    # Memory usage
    param_count = sum(param.data.size for param in model.parameters())
    memory_mb = param_count * 4 / (1024 * 1024)
    
    return {
        "forward_time_ms": forward_time * 1000,
        "memory_usage_mb": memory_mb,
        "parameter_count": param_count
    }


if __name__ == "__main__":
    print("🖼️  TensorBrain Computer Vision Layers")
    print("=" * 40)
    
    # Create CNN model
    cnn = create_cnn_model(input_channels=3, num_classes=10)
    print(f"CNN Model: {cnn}")
    print(f"Parameters: {sum(param.data.size for param in cnn.parameters()):,}")
    
    # Create sample data
    image_data = create_sample_image_data(batch_size=10, channels=3, height=32, width=32)
    print(f"Created {len(image_data)} sample images")
    
    # Test forward pass
    print("\n🧪 Testing CNN forward pass...")
    sample_image, sample_label = image_data[0]
    image_batch = Tensor(sample_image.data.reshape(1, *sample_image.shape), requires_grad=False)
    output = cnn(image_batch)
    print(f"Input shape: {image_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output.data[0, :5]}")  # Show first 5 outputs
    
    # Benchmark CNN
    benchmark_results = benchmark_cnn(cnn, image_data)
    
    print("\n📊 CNN Benchmark Results:")
    print(f"Forward time: {benchmark_results['forward_time_ms']:.2f}ms")
    print(f"Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    print(f"Parameter count: {benchmark_results['parameter_count']:,}")
    
    print("\n🎉 Computer Vision layers are working!")
    print("📝 Next steps:")
    print("   • Add BatchNorm2D")
    print("   • Implement Dropout2D")
    print("   • Add more activation functions")
    print("   • Implement residual connections")
    print("   • Add data augmentation")
    print("   • Train on real image datasets")
