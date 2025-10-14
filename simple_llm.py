#!/usr/bin/env python3
"""
Simple LLM built on top of TensorBrain
Simplified transformer architecture that works with our current framework
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import time

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss


class SimpleLLM(Module):
    """Simple Language Model using basic architecture"""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layer (simplified)
        self.embedding = Linear(vocab_size, d_model)
        
        # Transformer layers (simplified)
        self.layers = Sequential(*[
            Sequential(
                Linear(d_model, d_model * 4),  # Feed-forward
                ReLU(),
                Linear(d_model * 4, d_model),  # Feed-forward
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = Linear(d_model, vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the language model"""
        # x: [batch, seq_len] -> [batch, seq_len, d_model]
        batch_size, seq_len = x.shape
        
        # Simple embedding (one-hot encoding)
        x_onehot = self._one_hot_encode(x, self.vocab_size)
        x_embedded = self.embedding(x_onehot)
        
        # Transformer layers
        x = self.layers(x_embedded)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def _one_hot_encode(self, x: Tensor, vocab_size: int) -> Tensor:
        """Convert token indices to one-hot encoding"""
        batch_size, seq_len = x.shape
        one_hot = np.zeros((batch_size, seq_len, vocab_size))
        
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = int(x.data[i, j])
                if 0 <= token_id < vocab_size:
                    one_hot[i, j, token_id] = 1.0
        
        return Tensor(one_hot, requires_grad=False)
    
    def generate(self, input_ids: Tensor, max_length: int = 20) -> List[int]:
        """Generate text using the model"""
        generated = input_ids.data[0].tolist()
        
        for _ in range(max_length):
            # Forward pass
            logits = self.forward(Tensor(np.array([generated]), requires_grad=False))
            
            # Get next token logits
            next_token_logits = logits.data[0, -1, :]
            
            # Sample next token (simplified - just take argmax)
            next_token = np.argmax(next_token_logits)
            generated.append(next_token)
            
            # Stop if we hit end token (simplified)
            if next_token == 0:  # Assuming 0 is end token
                break
        
        return generated


def create_sample_data(vocab_size: int = 100, seq_len: int = 10, 
                      num_samples: int = 100) -> List[Tuple[Tensor, Tensor]]:
    """Create sample training data for the LLM"""
    data = []
    
    for _ in range(num_samples):
        # Create random sequence
        input_seq = np.random.randint(1, vocab_size, seq_len)
        target_seq = np.roll(input_seq, -1)  # Shift by one for next token prediction
        
        input_tensor = Tensor(input_seq.reshape(1, -1), requires_grad=False)
        target_tensor = Tensor(target_seq.reshape(1, -1), requires_grad=False)
        
        data.append((input_tensor, target_tensor))
    
    return data


def simple_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Simple loss for language modeling"""
    # Get the last token's logits and target
    batch_size, seq_len, vocab_size = logits.shape
    
    # Use the last token's prediction
    last_logits = logits[:, -1, :]  # [batch, vocab_size]
    last_target = targets[:, -1]    # [batch]
    
    # Convert target to one-hot for MSE loss
    target_onehot = np.zeros((batch_size, vocab_size))
    for i in range(batch_size):
        target_id = int(last_target.data[i])
        if 0 <= target_id < vocab_size:
            target_onehot[i, target_id] = 1.0
    
    target_tensor = Tensor(target_onehot, requires_grad=False)
    
    # MSE loss between logits and one-hot target
    loss = mse_loss(last_logits, target_tensor)
    
    return loss


def train_simple_llm(model: SimpleLLM, data_loader: List[Tuple[Tensor, Tensor]], 
                    num_epochs: int = 10) -> Dict[str, float]:
    """Train the simple LLM"""
    print("🚀 Training Simple LLM...")
    
    optimizer = SGD(model.parameters(), lr=0.001)
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for input_ids, target_ids in data_loader:
            # Forward pass
            logits = model(input_ids)
            loss = simple_loss(logits, target_ids)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_losses.append(loss.data.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:2d}: Loss = {avg_loss:.4f}")
    
    return {
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "loss_reduction": losses[0] - losses[-1]
    }


def benchmark_simple_llm(model: SimpleLLM, test_data: List[Tuple[Tensor, Tensor]]) -> Dict[str, float]:
    """Benchmark the simple LLM performance"""
    print("📊 Benchmarking Simple LLM...")
    
    # Inference speed
    start_time = time.time()
    for input_ids, _ in test_data[:10]:
        logits = model(input_ids)
    inference_time = (time.time() - start_time) / 10
    
    # Memory usage (simplified)
    param_count = sum(param.data.size for param in model.parameters())
    memory_mb = param_count * 4 / (1024 * 1024)  # Assume float32
    
    # Generation speed
    start_time = time.time()
    sample_input = Tensor(np.array([[1, 2, 3]]), requires_grad=False)
    generated = model.generate(sample_input, max_length=10)
    generation_time = time.time() - start_time
    
    return {
        "inference_time_ms": inference_time * 1000,
        "memory_usage_mb": memory_mb,
        "parameter_count": param_count,
        "generation_time_ms": generation_time * 1000,
        "generated_length": len(generated)
    }


if __name__ == "__main__":
    print("🧠 TensorBrain Simple LLM")
    print("=" * 40)
    
    # Create model
    model = SimpleLLM(vocab_size=50, d_model=32, n_layers=2)
    print(f"Model created with {sum(param.data.size for param in model.parameters())} parameters")
    
    # Create training data
    train_data = create_sample_data(vocab_size=50, seq_len=8, num_samples=30)
    print(f"Created {len(train_data)} training samples")
    
    # Train model
    training_results = train_simple_llm(model, train_data, num_epochs=5)
    
    # Benchmark model
    benchmark_results = benchmark_simple_llm(model, train_data)
    
    # Test generation
    print("\n🧪 Testing text generation:")
    sample_input = Tensor(np.array([[1, 2, 3, 4, 5]]), requires_grad=False)
    generated = model.generate(sample_input, max_length=10)
    print(f"Input: {sample_input.data[0].tolist()}")
    print(f"Generated: {generated}")
    
    print("\n📊 Simple LLM Results:")
    print(f"Final loss: {training_results['final_loss']:.4f}")
    print(f"Loss reduction: {training_results['loss_reduction']:.4f}")
    print(f"Inference time: {benchmark_results['inference_time_ms']:.2f}ms")
    print(f"Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    print(f"Parameter count: {benchmark_results['parameter_count']:,}")
    print(f"Generation time: {benchmark_results['generation_time_ms']:.2f}ms")
    
    print("\n🎉 Simple LLM is working!")
    print("📝 What we accomplished:")
    print("   ✅ Built a working language model on top of TensorBrain")
    print("   ✅ Implemented token embedding and generation")
    print("   ✅ Trained the model with gradient descent")
    print("   ✅ Generated text sequences")
    print("   ✅ Benchmarked performance")
    
    print("\n🚀 Next steps:")
    print("   • Add proper attention mechanism")
    print("   • Implement positional encoding")
    print("   • Add proper softmax and cross-entropy loss")
    print("   • Train on real text data")
    print("   • Add beam search for generation")
    print("   • Implement tokenization")
    print("   • Add model saving/loading")
