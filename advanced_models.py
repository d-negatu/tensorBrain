#!/usr/bin/env python3
"""
Advanced Model Architectures for TensorBrain
ResNet, Transformer, and other state-of-the-art models
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
import time

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss
from cv import Conv2D, MaxPool2D, AvgPool2D, Flatten


class BatchNorm2D(Module):
    """2D Batch Normalization"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Tensor(np.ones(num_features), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features), requires_grad=True)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through batch norm"""
        # x: [batch, channels, height, width]
        batch_size, channels, height, width = x.shape
        
        if self.training:
            # Calculate batch statistics
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.flatten()
        else:
            # Use running statistics
            mean = Tensor(self.running_mean.reshape(1, -1, 1, 1), requires_grad=False)
            var = Tensor(self.running_var.reshape(1, -1, 1, 1), requires_grad=False)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        weight = self.weight.data.reshape(1, -1, 1, 1)
        bias = self.bias.data.reshape(1, -1, 1, 1)
        
        output = x_norm * weight + bias
        
        return Tensor(output, requires_grad=x.requires_grad)
    
    def eval(self):
        """Set to evaluation mode"""
        self.training = False
    
    def train(self):
        """Set to training mode"""
        self.training = True


class ResidualBlock(Module):
    """Residual block for ResNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2D(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm2D(out_channels)
            )
        else:
            self.shortcut = None
        
        self.relu2 = ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through residual block"""
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        
        # Add residual connection
        out = out + shortcut
        out = self.relu2(out)
        
        return out


class ResNet(Module):
    """ResNet architecture"""
    
    def __init__(self, num_classes: int = 10, layers: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2D(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Classifier
        self.avgpool = AvgPool2D(kernel_size=7, stride=1)
        self.flatten = Flatten()
        self.fc = Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> Sequential:
        """Make a layer with multiple residual blocks"""
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ResNet"""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classifier
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


class MultiHeadAttention(Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through multi-head attention"""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attention_weights = self._softmax(scores)
        attention_output = attention_weights @ V
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output
    
    def _softmax(self, x: Tensor) -> Tensor:
        """Softmax function"""
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(softmax_x, requires_grad=x.requires_grad)


class TransformerBlock(Module):
    """Transformer block with attention and feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = Sequential(
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model)
        )
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block"""
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class LayerNorm(Module):
    """Layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through layer norm"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = x_norm * self.gamma + self.beta
        
        return output


class Transformer(Module):
    """Complete Transformer model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, max_seq_len: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = Linear(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = Sequential(*[
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = Linear(d_model, vocab_size)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Create positional encoding"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * 
                         (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return Tensor(pe, requires_grad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer"""
        # x: [batch, seq_len] -> [batch, seq_len, d_model]
        batch_size, seq_len = x.shape
        
        # Token embeddings (simplified)
        x_embed = self.embedding(x.reshape(-1, self.vocab_size))
        x_embed = x_embed.reshape(batch_size, seq_len, self.d_model)
        
        # Add positional encoding
        x = x_embed + self.pos_encoding.data[:seq_len, :]
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits


def benchmark_advanced_models():
    """Benchmark advanced model architectures"""
    print("📊 Benchmarking Advanced Model Architectures...")
    
    results = {}
    
    # ResNet benchmark
    print("🔄 Benchmarking ResNet...")
    resnet = ResNet(num_classes=10, layers=[2, 2, 2, 2])
    resnet_input = Tensor(np.random.randn(1, 3, 224, 224), requires_grad=False)
    
    start_time = time.time()
    resnet_output = resnet(resnet_input)
    resnet_time = (time.time() - start_time) * 1000
    
    results["resnet"] = {
        "forward_time_ms": resnet_time,
        "parameters": sum(param.data.size for param in resnet.parameters()),
        "output_shape": resnet_output.shape
    }
    
    # Transformer benchmark
    print("🔄 Benchmarking Transformer...")
    transformer = Transformer(vocab_size=1000, d_model=512, n_heads=8, n_layers=6)
    transformer_input = Tensor(np.random.randint(0, 1000, (1, 50)), requires_grad=False)
    
    start_time = time.time()
    transformer_output = transformer(transformer_input)
    transformer_time = (time.time() - start_time) * 1000
    
    results["transformer"] = {
        "forward_time_ms": transformer_time,
        "parameters": sum(param.data.size for param in transformer.parameters()),
        "output_shape": transformer_output.shape
    }
    
    print(f"\n📊 Advanced Models Benchmark Results:")
    print(f"ResNet:")
    print(f"  Forward time: {results['resnet']['forward_time_ms']:.2f}ms")
    print(f"  Parameters: {results['resnet']['parameters']:,}")
    print(f"  Output shape: {results['resnet']['output_shape']}")
    
    print(f"Transformer:")
    print(f"  Forward time: {results['transformer']['forward_time_ms']:.2f}ms")
    print(f"  Parameters: {results['transformer']['parameters']:,}")
    print(f"  Output shape: {results['transformer']['output_shape']}")
    
    return results


if __name__ == "__main__":
    print("🏗️  TensorBrain Advanced Model Architectures")
    print("=" * 50)
    
    # Benchmark advanced models
    benchmark_results = benchmark_advanced_models()
    
    print("\n🎉 Advanced model architectures are working!")
    print("📝 Next steps:")
    print("   • Add more architectures (BERT, GPT, etc.)")
    print("   • Implement attention mechanisms")
    print("   • Add positional encoding")
    print("   • Implement layer normalization")
    print("   • Add dropout for regularization")
    print("   • Implement model parallelism")
    print("   • Add gradient checkpointing")
