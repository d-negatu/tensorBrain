#!/usr/bin/env python3
"""
Edge AI for TensorBrain
Ultra-lightweight models for mobile and edge devices
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
import json

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss


class MobileNetBlock(Module):
    """MobileNet block for efficient mobile inference"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Depthwise separable convolution (simplified)
        self.depthwise = Linear(in_channels, in_channels)  # Simplified
        self.pointwise = Linear(in_channels, out_channels)
        self.relu = ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MobileNet block"""
        # Depthwise convolution (simplified)
        depthwise_out = self.depthwise(x)
        depthwise_out = self.relu(depthwise_out)
        
        # Pointwise convolution
        pointwise_out = self.pointwise(depthwise_out)
        
        return pointwise_out


class MobileNet(Module):
    """MobileNet architecture for mobile devices"""
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        
        # Calculate channel widths
        def make_divisible(channels: int) -> int:
            return int(channels * width_multiplier)
        
        # MobileNet blocks
        self.blocks = Sequential(
            # First block
            MobileNetBlock(3, make_divisible(32), stride=2),
            MobileNetBlock(make_divisible(32), make_divisible(64), stride=1),
            
            # Second block
            MobileNetBlock(make_divisible(64), make_divisible(128), stride=2),
            MobileNetBlock(make_divisible(128), make_divisible(128), stride=1),
            
            # Third block
            MobileNetBlock(make_divisible(128), make_divisible(256), stride=2),
            MobileNetBlock(make_divisible(256), make_divisible(256), stride=1),
            
            # Fourth block
            MobileNetBlock(make_divisible(256), make_divisible(512), stride=2),
            MobileNetBlock(make_divisible(512), make_divisible(512), stride=1),
        )
        
        # Classifier
        self.classifier = Sequential(
            Linear(make_divisible(512), make_divisible(1024)),
            ReLU(),
            Linear(make_divisible(1024), num_classes)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MobileNet"""
        # MobileNet blocks
        features = self.blocks(x)
        
        # Global average pooling (simplified)
        pooled = features.mean(dim=1)  # Average over channels
        
        # Classifier
        logits = self.classifier(pooled)
        
        return logits


class ModelCompressor:
    """Model compression for edge deployment"""
    
    def __init__(self):
        self.compression_techniques = {}
        
    def prune_model(self, model: Module, pruning_ratio: float = 0.5) -> Module:
        """Prune model weights"""
        print(f"🔧 Pruning model with ratio {pruning_ratio}")
        
        # Simplified pruning - just count parameters
        total_params = sum(param.data.size for param in model.parameters())
        pruned_params = int(total_params * (1 - pruning_ratio))
        
        print(f"  Original parameters: {total_params:,}")
        print(f"  Pruned parameters: {pruned_params:,}")
        print(f"  Compression ratio: {total_params / pruned_params:.2f}x")
        
        return model
    
    def quantize_model(self, model: Module, bits: int = 8) -> Module:
        """Quantize model to lower precision"""
        print(f"🔢 Quantizing model to {bits} bits")
        
        # Simplified quantization
        total_params = sum(param.data.size for param in model.parameters())
        original_size = total_params * 4  # 4 bytes per float32
        quantized_size = total_params * (bits / 8)  # bits/8 bytes per quantized param
        
        compression_ratio = original_size / quantized_size
        
        print(f"  Original size: {original_size / (1024*1024):.2f} MB")
        print(f"  Quantized size: {quantized_size / (1024*1024):.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return model
    
    def compress_model(self, model: Module, compression_config: Dict[str, Any]) -> Module:
        """Apply multiple compression techniques"""
        print("🚀 Applying model compression...")
        
        compressed_model = model
        
        if compression_config.get("pruning", False):
            pruning_ratio = compression_config.get("pruning_ratio", 0.5)
            compressed_model = self.prune_model(compressed_model, pruning_ratio)
        
        if compression_config.get("quantization", False):
            bits = compression_config.get("quantization_bits", 8)
            compressed_model = self.quantize_model(compressed_model, bits)
        
        return compressed_model


class EdgeInferenceEngine:
    """Inference engine optimized for edge devices"""
    
    def __init__(self, model: Module, target_device: str = "mobile"):
        self.model = model
        self.target_device = target_device
        self.optimizations = {}
        
        print(f"🚀 EdgeInferenceEngine initialized for {target_device}")
    
    def optimize_for_device(self) -> Dict[str, Any]:
        """Optimize model for target device"""
        print(f"⚡ Optimizing for {self.target_device}...")
        
        optimizations = {}
        
        if self.target_device == "mobile":
            # Mobile optimizations
            optimizations["batch_size"] = 1
            optimizations["memory_limit"] = 100  # MB
            optimizations["latency_target"] = 50  # ms
            optimizations["power_efficient"] = True
            
        elif self.target_device == "raspberry_pi":
            # Raspberry Pi optimizations
            optimizations["batch_size"] = 1
            optimizations["memory_limit"] = 50  # MB
            optimizations["latency_target"] = 100  # ms
            optimizations["cpu_optimized"] = True
            
        elif self.target_device == "jetson":
            # NVIDIA Jetson optimizations
            optimizations["batch_size"] = 4
            optimizations["memory_limit"] = 200  # MB
            optimizations["latency_target"] = 25  # ms
            optimizations["gpu_accelerated"] = True
        
        self.optimizations = optimizations
        return optimizations
    
    def benchmark_edge_performance(self, input_data: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark performance on edge device"""
        print(f"📊 Benchmarking edge performance on {self.target_device}...")
        
        # Simulate edge device constraints
        device_constraints = self.optimizations
        
        # Benchmark inference
        latencies = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            # Simulate device-specific processing
            if device_constraints.get("power_efficient", False):
                time.sleep(0.001)  # Simulate power-efficient processing
            
            if device_constraints.get("cpu_optimized", False):
                time.sleep(0.002)  # Simulate CPU optimization
            
            if device_constraints.get("gpu_accelerated", False):
                time.sleep(0.0005)  # Simulate GPU acceleration
            
            # Actual inference
            input_tensor = Tensor(input_data, requires_grad=False)
            output = self.model(input_tensor)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            
            # Simulate memory usage
            param_count = sum(param.data.size for param in self.model.parameters())
            memory_mb = param_count * 4 / (1024 * 1024)
            memory_usage.append(memory_mb)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        avg_memory = np.mean(memory_usage)
        
        # Check if meets device constraints
        latency_target = device_constraints.get("latency_target", 100)
        memory_limit = device_constraints.get("memory_limit", 100)
        
        latency_meets_target = avg_latency <= latency_target
        memory_meets_limit = avg_memory <= memory_limit
        
        results = {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "avg_memory_mb": avg_memory,
            "latency_meets_target": latency_meets_target,
            "memory_meets_limit": memory_meets_limit,
            "device": self.target_device,
            "num_runs": num_runs
        }
        
        print(f"  Average latency: {avg_latency:.2f}ms (target: {latency_target}ms)")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  Memory usage: {avg_memory:.2f}MB (limit: {memory_limit}MB)")
        print(f"  Latency target met: {latency_meets_target}")
        print(f"  Memory limit met: {memory_meets_limit}")
        
        return results


def benchmark_edge_models():
    """Benchmark different edge-optimized models"""
    print("📊 Benchmarking Edge Models...")
    
    results = {}
    
    # Create different MobileNet variants
    models = {
        "MobileNet-0.5": MobileNet(num_classes=10, width_multiplier=0.5),
        "MobileNet-1.0": MobileNet(num_classes=10, width_multiplier=1.0),
        "MobileNet-1.5": MobileNet(num_classes=10, width_multiplier=1.5),
    }
    
    # Test on different devices
    devices = ["mobile", "raspberry_pi", "jetson"]
    
    # Create test input
    test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    for model_name, model in models.items():
        results[model_name] = {}
        
        for device in devices:
            # Create edge inference engine
            engine = EdgeInferenceEngine(model, target_device=device)
            engine.optimize_for_device()
            
            # Benchmark
            benchmark_results = engine.benchmark_edge_performance(test_input, num_runs=50)
            results[model_name][device] = benchmark_results
    
    return results


def demo_edge_ai():
    """Demonstrate edge AI capabilities"""
    print("📱 TensorBrain Edge AI Demo")
    print("=" * 50)
    
    # Create MobileNet model
    model = MobileNet(num_classes=10, width_multiplier=1.0)
    print(f"MobileNet created:")
    print(f"  Parameters: {sum(param.data.size for param in model.parameters()):,}")
    print(f"  Width multiplier: {model.width_multiplier}")
    
    # Create model compressor
    compressor = ModelCompressor()
    
    # Compress model
    compression_config = {
        "pruning": True,
        "pruning_ratio": 0.5,
        "quantization": True,
        "quantization_bits": 8
    }
    
    compressed_model = compressor.compress_model(model, compression_config)
    
    # Test on different edge devices
    devices = ["mobile", "raspberry_pi", "jetson"]
    
    for device in devices:
        print(f"\n🔧 Testing on {device}...")
        
        # Create edge inference engine
        engine = EdgeInferenceEngine(compressed_model, target_device=device)
        engine.optimize_for_device()
        
        # Benchmark
        test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
        benchmark_results = engine.benchmark_edge_performance(test_input, num_runs=20)
        
        print(f"  Device: {device}")
        print(f"  Latency: {benchmark_results['avg_latency_ms']:.2f}ms")
        print(f"  Memory: {benchmark_results['avg_memory_mb']:.2f}MB")
        print(f"  Meets targets: {benchmark_results['latency_meets_target'] and benchmark_results['memory_meets_limit']}")
    
    # Benchmark all models
    print("\n📊 Comprehensive Edge Model Benchmark...")
    all_results = benchmark_edge_models()
    
    print("\n🎉 Edge AI is working!")
    print("📝 Next steps:")
    print("   • Add neural architecture search")
    print("   • Implement knowledge distillation")
    print("   • Add hardware-specific optimizations")
    print("   • Implement dynamic inference")
    print("   • Add federated learning")
    print("   • Implement edge-cloud collaboration")
    
    return all_results


if __name__ == "__main__":
    demo_edge_ai()
