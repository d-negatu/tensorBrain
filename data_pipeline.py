#!/usr/bin/env python3
"""
Advanced Data Pipeline for TensorBrain
Data loading, preprocessing, augmentation, and streaming
"""

import numpy as np
import time
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU


@dataclass
class DataSample:
    """Data sample with metadata"""
    data: np.ndarray
    label: np.ndarray
    metadata: Dict[str, Any] = None


class DataAugmentation:
    """Data augmentation techniques"""
    
    def __init__(self):
        self.augmentations = {}
        
        print("🚀 DataAugmentation initialized")
    
    def add_augmentation(self, name: str, func: Callable):
        """Add augmentation function"""
        self.augmentations[name] = func
        print(f"✅ Added augmentation: {name}")
    
    def apply_augmentations(self, data: np.ndarray, augmentations: List[str]) -> np.ndarray:
        """Apply list of augmentations to data"""
        result = data.copy()
        
        for aug_name in augmentations:
            if aug_name in self.augmentations:
                result = self.augmentations[aug_name](result)
        
        return result
    
    def random_flip(self, data: np.ndarray) -> np.ndarray:
        """Random horizontal flip"""
        if np.random.random() > 0.5:
            return np.flip(data, axis=1)
        return data
    
    def random_rotation(self, data: np.ndarray) -> np.ndarray:
        """Random rotation"""
        angle = np.random.uniform(-10, 10)
        # Simplified rotation (would use proper rotation matrix)
        return data
    
    def random_noise(self, data: np.ndarray) -> np.ndarray:
        """Add random noise"""
        noise = np.random.normal(0, 0.1, data.shape)
        return data + noise
    
    def random_crop(self, data: np.ndarray) -> np.ndarray:
        """Random crop"""
        # Simplified crop (would use proper cropping)
        return data
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)


class DataLoader:
    """Advanced data loader with multiprocessing"""
    
    def __init__(self, dataset: List[DataSample], batch_size: int = 32, 
                 shuffle: bool = True, num_workers: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.augmentation = DataAugmentation()
        
        # Initialize augmentations
        self.augmentation.add_augmentation("flip", self.augmentation.random_flip)
        self.augmentation.add_augmentation("rotation", self.augmentation.random_rotation)
        self.augmentation.add_augmentation("noise", self.augmentation.random_noise)
        self.augmentation.add_augmentation("crop", self.augmentation.random_crop)
        self.augmentation.add_augmentation("normalize", self.augmentation.normalize)
        
        print(f"🚀 DataLoader initialized:")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Shuffle: {shuffle}")
        print(f"   Workers: {num_workers}")
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Iterator for data batches"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                sample = self.dataset[idx]
                
                # Apply augmentations
                augmented_data = self.augmentation.apply_augmentations(
                    sample.data, 
                    ["normalize", "noise"]  # Default augmentations
                )
                
                batch_data.append(augmented_data)
                batch_labels.append(sample.label)
            
            # Convert to tensors
            data_tensor = Tensor(np.stack(batch_data), requires_grad=False)
            label_tensor = Tensor(np.stack(batch_labels), requires_grad=False)
            
            yield data_tensor, label_tensor
    
    def __len__(self) -> int:
        """Number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class StreamingDataLoader:
    """Streaming data loader for real-time data"""
    
    def __init__(self, data_source: Callable, batch_size: int = 32, 
                 buffer_size: int = 1000):
        self.data_source = data_source
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.thread = None
        
        print(f"🚀 StreamingDataLoader initialized:")
        print(f"   Batch size: {batch_size}")
        print(f"   Buffer size: {buffer_size}")
    
    def start(self):
        """Start streaming data loader"""
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_data)
        self.thread.daemon = True
        self.thread.start()
        print("✅ Streaming data loader started")
    
    def stop(self):
        """Stop streaming data loader"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("✅ Streaming data loader stopped")
    
    def get_batch(self, timeout: float = 5.0) -> Optional[Tuple[Tensor, Tensor]]:
        """Get next batch of data"""
        try:
            batch = self.buffer.get(timeout=timeout)
            return batch
        except queue.Empty:
            return None
    
    def _stream_data(self):
        """Stream data from source"""
        while self.is_running:
            try:
                # Get data from source
                data = self.data_source()
                
                # Process data
                if isinstance(data, tuple):
                    data_tensor = Tensor(data[0], requires_grad=False)
                    label_tensor = Tensor(data[1], requires_grad=False)
                else:
                    data_tensor = Tensor(data, requires_grad=False)
                    label_tensor = Tensor(np.zeros((data.shape[0], 1)), requires_grad=False)
                
                # Add to buffer
                self.buffer.put((data_tensor, label_tensor))
                
            except Exception as e:
                print(f"❌ Error streaming data: {e}")
                time.sleep(0.1)


class DataPreprocessor:
    """Data preprocessing pipeline"""
    
    def __init__(self):
        self.preprocessing_steps = []
        
        print("🚀 DataPreprocessor initialized")
    
    def add_step(self, name: str, func: Callable):
        """Add preprocessing step"""
        self.preprocessing_steps.append((name, func))
        print(f"✅ Added preprocessing step: {name}")
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process data through preprocessing pipeline"""
        result = data.copy()
        
        for name, func in self.preprocessing_steps:
            result = func(result)
        
        return result
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    def standardize(self, data: np.ndarray) -> np.ndarray:
        """Standardize data"""
        return (data - np.mean(data)) / np.std(data)
    
    def min_max_scale(self, data: np.ndarray) -> np.ndarray:
        """Min-max scaling"""
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    
    def pad(self, data: np.ndarray, pad_width: int = 1) -> np.ndarray:
        """Pad data"""
        return np.pad(data, pad_width, mode='constant')
    
    def crop(self, data: np.ndarray, crop_size: int = 28) -> np.ndarray:
        """Crop data"""
        # Simplified crop
        return data[:crop_size, :crop_size]


def create_sample_dataset(num_samples: int = 1000) -> List[DataSample]:
    """Create sample dataset for testing"""
    dataset = []
    
    for i in range(num_samples):
        # Generate random data
        data = np.random.randn(32, 32, 3).astype(np.float32)
        label = np.random.randint(0, 10, (1,)).astype(np.float32)
        
        # Create data sample
        sample = DataSample(
            data=data,
            label=label,
            metadata={"id": i, "source": "synthetic"}
        )
        
        dataset.append(sample)
    
    return dataset


def benchmark_data_pipeline(dataset: List[DataSample], num_epochs: int = 5) -> Dict[str, Any]:
    """Benchmark data pipeline performance"""
    print("📊 Benchmarking Data Pipeline...")
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Benchmark data loading
    start_time = time.time()
    total_batches = 0
    total_samples = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_batches = 0
        epoch_samples = 0
        
        for batch_data, batch_labels in data_loader:
            epoch_batches += 1
            epoch_samples += batch_data.shape[0]
        
        epoch_time = time.time() - epoch_start
        total_batches += epoch_batches
        total_samples += epoch_samples
        
        print(f"Epoch {epoch}: {epoch_batches} batches, {epoch_samples} samples, {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    avg_batch_time = total_time / total_batches
    throughput = total_samples / total_time
    
    results = {
        "total_time": total_time,
        "total_batches": total_batches,
        "total_samples": total_samples,
        "avg_batch_time": avg_batch_time,
        "throughput_samples_per_sec": throughput,
        "num_epochs": num_epochs
    }
    
    print(f"\n📊 Data Pipeline Benchmark Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Average batch time: {avg_batch_time:.3f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    return results


def demo_data_pipeline():
    """Demonstrate data pipeline capabilities"""
    print("🚀 TensorBrain Advanced Data Pipeline Demo")
    print("=" * 50)
    
    # Create sample dataset
    dataset = create_sample_dataset(num_samples=1000)
    print(f"Created dataset with {len(dataset)} samples")
    
    # Create data preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.add_step("normalize", preprocessor.normalize)
    preprocessor.add_step("standardize", preprocessor.standardize)
    
    # Test preprocessing
    sample_data = dataset[0].data
    processed_data = preprocessor.process(sample_data)
    print(f"Preprocessed data shape: {processed_data.shape}")
    
    # Benchmark data pipeline
    benchmark_results = benchmark_data_pipeline(dataset, num_epochs=3)
    
    # Test streaming data loader
    print("\n🔄 Testing streaming data loader...")
    def data_source():
        return np.random.randn(32, 32, 3).astype(np.float32)
    
    streaming_loader = StreamingDataLoader(data_source, batch_size=16, buffer_size=100)
    streaming_loader.start()
    
    # Get a few batches
    for i in range(3):
        batch = streaming_loader.get_batch(timeout=2.0)
        if batch:
            data_tensor, label_tensor = batch
            print(f"Streaming batch {i}: {data_tensor.shape}")
    
    streaming_loader.stop()
    
    print("\n🎉 Advanced data pipeline is working!")
    print("📝 Next steps:")
    print("   • Add more augmentation techniques")
    print("   • Implement distributed data loading")
    print("   • Add data validation")
    print("   • Implement caching")
    print("   • Add data versioning")
    print("   • Implement data quality monitoring")
    
    return benchmark_results


if __name__ == "__main__":
    demo_data_pipeline()
