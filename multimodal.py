#!/usr/bin/env python3
"""
Multi-Modal AI for TensorBrain
Vision + Language models that can see and understand
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
import json

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss
from cv import Conv2D, MaxPool2D, Flatten
from real_llm import RealLLM, Tokenizer


class VisionEncoder(Module):
    """Vision encoder for processing images"""
    
    def __init__(self, input_channels: int = 3, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # CNN backbone
        self.cnn_backbone = Sequential(
            Conv2D(input_channels, 64, kernel_size=7, stride=2, padding=3),
            ReLU(),
            MaxPool2D(kernel_size=3, stride=2, padding=1),
            
            Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            
            Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            
            Conv2D(256, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
        )
        
        # Projection to d_model
        self.projection = Linear(512 * 4 * 4, d_model)  # Assuming 32x32 -> 4x4
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through vision encoder"""
        # x: [batch, channels, height, width]
        batch_size = x.shape[0]
        
        # CNN features
        cnn_features = self.cnn_backbone(x)
        
        # Flatten and project
        flattened = cnn_features.reshape(batch_size, -1)
        projected = self.projection(flattened)
        
        return projected


class LanguageEncoder(Module):
    """Language encoder for processing text"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, max_seq_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = Linear(vocab_size, d_model)
        
        # Language model layers
        self.language_layers = Sequential(
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through language encoder"""
        # x: [batch, seq_len, vocab_size] (one-hot)
        batch_size, seq_len, vocab_size = x.shape
        
        # Embed tokens
        x_flat = x.reshape(batch_size * seq_len, vocab_size)
        embedded = self.embedding(x_flat)
        embedded = embedded.reshape(batch_size, seq_len, self.d_model)
        
        # Language processing
        # Average pool over sequence length
        pooled = embedded.mean(dim=1)  # [batch, d_model]
        
        # Language layers
        output = self.language_layers(pooled)
        
        return output


class CrossModalAttention(Module):
    """Cross-modal attention between vision and language"""
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Attention projections
        self.vision_proj = Linear(d_model, d_model)
        self.language_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
    def forward(self, vision_features: Tensor, language_features: Tensor) -> Tensor:
        """Cross-modal attention"""
        # vision_features: [batch, d_model]
        # language_features: [batch, d_model]
        
        # Project features
        vision_proj = self.vision_proj(vision_features)
        language_proj = self.language_proj(language_features)
        
        # Compute attention scores
        attention_scores = vision_proj * language_proj  # Element-wise
        
        # Apply attention
        attended_vision = vision_features * attention_scores
        attended_language = language_features * attention_scores
        
        # Combine features
        combined = attended_vision + attended_language
        
        # Output projection
        output = self.output_proj(combined)
        
        return output


class MultiModalModel(Module):
    """Complete multi-modal model"""
    
    def __init__(self, vocab_size: int, num_classes: int = 10, d_model: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Encoders
        self.vision_encoder = VisionEncoder(input_channels=3, d_model=d_model)
        self.language_encoder = LanguageEncoder(vocab_size=vocab_size, d_model=d_model)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(d_model=d_model)
        
        # Classification head
        self.classifier = Sequential(
            Linear(d_model, d_model // 2),
            ReLU(),
            Linear(d_model // 2, num_classes)
        )
        
    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        """Forward pass through multi-modal model"""
        # Encode modalities
        vision_features = self.vision_encoder(image)
        language_features = self.language_encoder(text)
        
        # Cross-modal attention
        fused_features = self.cross_attention(vision_features, language_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict(self, image: Tensor, text: Tensor) -> Dict[str, Any]:
        """Make prediction with confidence scores"""
        logits = self.forward(image, text)
        
        # Convert to probabilities (simplified)
        probs = np.exp(logits.data) / np.sum(np.exp(logits.data), axis=-1, keepdims=True)
        
        # Get predicted class
        predicted_class = np.argmax(probs, axis=-1)
        confidence = np.max(probs, axis=-1)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probs,
            "logits": logits.data
        }


class VisionLanguageDataset:
    """Dataset for vision-language tasks"""
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.samples = self._generate_samples()
        
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic vision-language samples"""
        samples = []
        
        for i in range(self.num_samples):
            # Generate random image
            image = np.random.randn(3, 32, 32).astype(np.float32)
            
            # Generate random text (one-hot)
            text_length = np.random.randint(5, 15)
            vocab_size = 100
            text = np.zeros((text_length, vocab_size), dtype=np.float32)
            for j in range(text_length):
                token_id = np.random.randint(0, vocab_size)
                text[j, token_id] = 1.0
            
            # Generate label
            label = np.random.randint(0, 10)
            
            samples.append({
                "image": image,
                "text": text,
                "label": label,
                "id": i
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
    
    def get_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get batch of samples"""
        batch = []
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        
        for idx in indices:
            batch.append(self.samples[idx])
        
        return batch


def train_multimodal_model(model: MultiModalModel, dataset: VisionLanguageDataset, 
                          num_epochs: int = 10) -> Dict[str, Any]:
    """Train multi-modal model"""
    print("🚀 Training Multi-Modal Model...")
    
    optimizer = SGD(model.parameters(), lr=0.01)
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Get batch
        batch = dataset.get_batch(batch_size=16)
        
        # Prepare batch data
        images = np.stack([sample["image"] for sample in batch])
        texts = np.stack([sample["text"] for sample in batch])
        labels = np.array([sample["label"] for sample in batch])
        
        # Convert to tensors
        image_tensor = Tensor(images, requires_grad=False)
        text_tensor = Tensor(texts, requires_grad=False)
        label_tensor = Tensor(labels, requires_grad=False)
        
        # Forward pass
        logits = model(image_tensor, text_tensor)
        
        # Compute loss (simplified)
        target_onehot = np.zeros((len(batch), 10), dtype=np.float32)
        for i, label in enumerate(labels):
            target_onehot[i, label] = 1.0
        
        target_tensor = Tensor(target_onehot, requires_grad=False)
        loss = mse_loss(logits, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_losses.append(loss.data.item())
        losses.append(np.mean(epoch_losses))
        
        print(f"Epoch {epoch}: Loss = {losses[-1]:.4f}")
    
    return {"final_loss": losses[-1], "losses": losses}


def benchmark_multimodal(model: MultiModalModel, dataset: VisionLanguageDataset) -> Dict[str, float]:
    """Benchmark multi-modal model"""
    print("📊 Benchmarking Multi-Modal Model...")
    
    # Test batch
    batch = dataset.get_batch(batch_size=8)
    
    # Prepare data
    images = np.stack([sample["image"] for sample in batch])
    texts = np.stack([sample["text"] for sample in batch])
    
    image_tensor = Tensor(images, requires_grad=False)
    text_tensor = Tensor(texts, requires_grad=False)
    
    # Benchmark inference
    start_time = time.time()
    predictions = model.predict(image_tensor, text_tensor)
    inference_time = (time.time() - start_time) * 1000
    
    # Calculate metrics
    param_count = sum(param.data.size for param in model.parameters())
    memory_mb = param_count * 4 / (1024 * 1024)
    
    return {
        "inference_time_ms": inference_time,
        "memory_usage_mb": memory_mb,
        "parameter_count": param_count,
        "batch_size": len(batch)
    }


def demo_multimodal():
    """Demonstrate multi-modal AI capabilities"""
    print("🤖 TensorBrain Multi-Modal AI Demo")
    print("=" * 50)
    
    # Create model
    model = MultiModalModel(vocab_size=100, num_classes=10, d_model=256)
    print(f"Multi-Modal Model created:")
    print(f"  Parameters: {sum(param.data.size for param in model.parameters()):,}")
    print(f"  Vision encoder: {model.vision_encoder}")
    print(f"  Language encoder: {model.language_encoder}")
    print(f"  Cross-modal attention: {model.cross_attention}")
    
    # Create dataset
    dataset = VisionLanguageDataset(num_samples=1000)
    print(f"Dataset created: {len(dataset)} samples")
    
    # Train model
    training_results = train_multimodal_model(model, dataset, num_epochs=5)
    
    # Benchmark
    benchmark_results = benchmark_multimodal(model, dataset)
    
    print(f"\n📊 Multi-Modal Benchmark Results:")
    print(f"Inference time: {benchmark_results['inference_time_ms']:.2f}ms")
    print(f"Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    print(f"Parameter count: {benchmark_results['parameter_count']:,}")
    
    # Test prediction
    print("\n🧪 Testing Multi-Modal Prediction:")
    sample = dataset[0]
    image = Tensor(sample["image"].reshape(1, *sample["image"].shape), requires_grad=False)
    text = Tensor(sample["text"].reshape(1, *sample["text"].shape), requires_grad=False)
    
    prediction = model.predict(image, text)
    print(f"Predicted class: {prediction['predicted_class'][0]}")
    print(f"Confidence: {prediction['confidence'][0]:.4f}")
    print(f"True label: {sample['label']}")
    
    print("\n🎉 Multi-Modal AI is working!")
    print("📝 Next steps:")
    print("   • Add attention mechanisms")
    print("   • Implement transformer-based fusion")
    print("   • Add more modalities (audio, video)")
    print("   • Implement contrastive learning")
    print("   • Add pre-trained models")
    print("   • Implement zero-shot learning")
    
    return benchmark_results


if __name__ == "__main__":
    demo_multimodal()
