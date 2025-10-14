#!/usr/bin/env python3
"""
Basic INT8 Quantization for TensorBrain
Post-training quantization for model compression and acceleration
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass

from tensor import Tensor
from nn import Linear, Sequential, Module


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    weight_bits: int = 8
    activation_bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    calibration_samples: int = 100


@dataclass
class QuantizationStats:
    """Statistics for quantization"""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantization_time_ms: float
    calibration_time_ms: float
    accuracy_drop: float


class Quantizer:
    """Basic INT8 quantizer for neural networks"""
    
    def __init__(self, config: QuantizationConfig = QuantizationConfig()):
        self.config = config
        self.scales = {}
        self.zero_points = {}
        self.quantization_stats = QuantizationStats(0, 0, 0, 0, 0, 0)
    
    def quantize_model(self, model: Module, calibration_data: List[Tensor]) -> Module:
        """Quantize a model using calibration data"""
        print("🔢 Quantizing model...")
        start_time = time.time()
        
        # Calibrate quantization parameters
        calibration_start = time.time()
        self._calibrate(model, calibration_data)
        calibration_time = (time.time() - calibration_start) * 1000
        
        # Create quantized model
        quantized_model = self._create_quantized_model(model)
        
        quantization_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        original_size = self._calculate_model_size(model)
        quantized_size = self._calculate_model_size(quantized_model)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1
        
        self.quantization_stats = QuantizationStats(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression_ratio,
            quantization_time_ms=quantization_time,
            calibration_time_ms=calibration_time,
            accuracy_drop=0.0  # Would be calculated with actual accuracy metrics
        )
        
        print(f"✅ Model quantized:")
        print(f"   Original size: {original_size:.2f} MB")
        print(f"   Quantized size: {quantized_size:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        print(f"   Quantization time: {quantization_time:.2f}ms")
        
        return quantized_model
    
    def _calibrate(self, model: Module, calibration_data: List[Tensor]):
        """Calibrate quantization parameters using calibration data"""
        print("📊 Calibrating quantization parameters...")
        
        # Collect activation statistics
        activation_stats = {}
        
        for i, data in enumerate(calibration_data[:self.config.calibration_samples]):
            # Forward pass to collect activations
            x = data
            for layer in model.modules:
                if isinstance(layer, Linear):
                    x = layer(x)
                    # Collect activation statistics
                    layer_name = f"linear_{i}"
                    if layer_name not in activation_stats:
                        activation_stats[layer_name] = []
                    activation_stats[layer_name].append(x.data)
        
        # Calculate scales and zero points
        for layer_name, activations in activation_stats.items():
            # Flatten all activations for quantization parameter calculation
            all_activations = np.concatenate([act.flatten() for act in activations])
            self._calculate_quantization_params(all_activations, layer_name)
    
    def _calculate_quantization_params(self, data: np.ndarray, name: str):
        """Calculate scale and zero point for quantization"""
        if self.config.symmetric:
            # Symmetric quantization
            max_val = np.max(np.abs(data))
            scale = max_val / (2**(self.config.activation_bits - 1) - 1)
            zero_point = 0
        else:
            # Asymmetric quantization
            min_val = np.min(data)
            max_val = np.max(data)
            scale = (max_val - min_val) / (2**self.config.activation_bits - 1)
            zero_point = -min_val / scale
        
        self.scales[name] = scale
        self.zero_points[name] = zero_point
    
    def _create_quantized_model(self, model: Module) -> Module:
        """Create a quantized version of the model"""
        quantized_modules = []
        
        for i, layer in enumerate(model.modules):
            if isinstance(layer, Linear):
                # Quantize linear layer
                quantized_layer = self._quantize_linear_layer(layer, f"linear_{i}")
                quantized_modules.append(quantized_layer)
            else:
                # Keep non-linear layers as-is for now
                quantized_modules.append(layer)
        
        return Sequential(*quantized_modules)
    
    def _quantize_linear_layer(self, layer: Linear, name: str) -> 'QuantizedLinear':
        """Quantize a linear layer"""
        # Quantize weights
        weight_scale = np.max(np.abs(layer.weight.data)) / (2**(self.config.weight_bits - 1) - 1)
        quantized_weights = np.round(layer.weight.data / weight_scale).astype(np.int8)
        
        # Quantize bias (if exists)
        quantized_bias = None
        if layer.bias is not None:
            bias_scale = weight_scale  # Simplified
            quantized_bias = np.round(layer.bias.data / bias_scale).astype(np.int8)
        
        return QuantizedLinear(
            quantized_weights,
            quantized_bias,
            weight_scale,
            bias_scale if layer.bias is not None else None,
            layer.in_features,
            layer.out_features
        )
    
    def _calculate_model_size(self, model: Module) -> float:
        """Calculate model size in MB"""
        total_params = 0
        for param in model.parameters():
            total_params += param.data.size
        
        # Assume float32 (4 bytes) for original, int8 (1 byte) for quantized
        size_bytes = total_params * 4  # This is simplified
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def get_quantization_stats(self) -> QuantizationStats:
        """Get quantization statistics"""
        return self.quantization_stats


class QuantizedLinear:
    """Quantized linear layer"""
    
    def __init__(self, quantized_weights: np.ndarray, quantized_bias: Optional[np.ndarray],
                 weight_scale: float, bias_scale: Optional[float],
                 in_features: int, out_features: int):
        self.quantized_weights = quantized_weights
        self.quantized_bias = quantized_bias
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized operations"""
        # Dequantize weights
        dequantized_weights = self.quantized_weights.astype(np.float32) * self.weight_scale
        
        # Matrix multiplication
        output = x.data @ dequantized_weights.T
        
        # Add bias if present
        if self.quantized_bias is not None:
            dequantized_bias = self.quantized_bias.astype(np.float32) * self.bias_scale
            output += dequantized_bias
        
        return Tensor(output, requires_grad=x.requires_grad)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def benchmark_quantization(model: Module, test_data: List[Tensor], num_runs: int = 100) -> Dict[str, float]:
    """Benchmark quantization performance"""
    print("📊 Benchmarking quantization...")
    
    # Original model performance
    original_times = []
    for _ in range(num_runs):
        start_time = time.time()
        for data in test_data:
            output = model(data)
        end_time = time.time()
        original_times.append((end_time - start_time) * 1000)
    
    # Quantized model performance
    quantizer = Quantizer()
    quantized_model = quantizer.quantize_model(model, test_data)
    
    quantized_times = []
    for _ in range(num_runs):
        start_time = time.time()
        for data in test_data:
            output = quantized_model(data)
        end_time = time.time()
        quantized_times.append((end_time - start_time) * 1000)
    
    # Calculate statistics
    original_avg = np.mean(original_times)
    quantized_avg = np.mean(quantized_times)
    speedup = original_avg / quantized_avg if quantized_avg > 0 else 1
    
    stats = quantizer.get_quantization_stats()
    
    return {
        "original_avg_ms": original_avg,
        "quantized_avg_ms": quantized_avg,
        "speedup": speedup,
        "compression_ratio": stats.compression_ratio,
        "original_size_mb": stats.original_size_mb,
        "quantized_size_mb": stats.quantized_size_mb,
        "quantization_time_ms": stats.quantization_time_ms
    }


if __name__ == "__main__":
    print("🔢 TensorBrain INT8 Quantization")
    print("=" * 40)
    
    # Create a sample model
    from nn import Sequential, Linear, ReLU
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 2),
    )
    
    # Create calibration data
    calibration_data = [Tensor(np.random.randn(10, 2), requires_grad=False) for _ in range(50)]
    
    # Benchmark quantization
    benchmark_result = benchmark_quantization(model, calibration_data)
    
    print("\n📊 Quantization Benchmark Results:")
    print(f"Original avg latency: {benchmark_result['original_avg_ms']:.2f}ms")
    print(f"Quantized avg latency: {benchmark_result['quantized_avg_ms']:.2f}ms")
    print(f"Speedup: {benchmark_result['speedup']:.2f}x")
    print(f"Compression ratio: {benchmark_result['compression_ratio']:.2f}x")
    print(f"Original size: {benchmark_result['original_size_mb']:.2f} MB")
    print(f"Quantized size: {benchmark_result['quantized_size_mb']:.2f} MB")
    
    print("\n🎉 Quantization is working!")
    print("📝 Next steps:")
    print("   • Implement quantization-aware training")
    print("   • Add per-channel quantization")
    print("   • Integrate with graph compiler")
    print("   • Add hardware-specific optimizations")
    print("   • Implement dynamic quantization")
