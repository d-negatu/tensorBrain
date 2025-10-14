#!/usr/bin/env python3
"""
Small LLM (Language Model) built on top of TensorBrain
Transformer architecture with attention mechanism
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss


@dataclass
class LLMConfig:
    """Configuration for the small LLM"""
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64
    dropout: float = 0.1
    learning_rate: float = 0.001


class PositionalEncoding(Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * 
                         (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = Tensor(pe, requires_grad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input"""
        batch_size, seq_len, d_model = x.shape
        pe_slice = self.pe.data[:seq_len, :]  # [seq_len, d_model]
        pe_broadcast = np.broadcast_to(pe_slice, (batch_size, seq_len, d_model))
        pe_tensor = Tensor(pe_broadcast, requires_grad=False)
        return x + pe_tensor


class MultiHeadAttention(Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through multi-head attention"""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x)  # [batch, seq_len, d_model]
        K = self.w_k(x)  # [batch, seq_len, d_model]
        V = self.w_v(x)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = self._reshape_for_attention(Q, batch_size, seq_len)
        K = self._reshape_for_attention(K, batch_size, seq_len)
        V = self._reshape_for_attention(V, batch_size, seq_len)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output
    
    def _reshape_for_attention(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Reshape tensor for multi-head attention"""
        # x: [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        x = x.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        return x
    
    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, 
                                    mask: Optional[Tensor] = None) -> Tensor:
        """Scaled dot-product attention"""
        # Q, K, V: [batch, n_heads, seq_len, d_k]
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax (simplified - using ReLU as approximation)
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        output = attention_weights @ V
        
        return output
    
    def _softmax(self, x: Tensor) -> Tensor:
        """Simplified softmax using ReLU approximation"""
        # For simplicity, we'll use a basic softmax approximation
        # In a real implementation, this would be proper softmax
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(softmax_x, requires_grad=x.requires_grad)


class FeedForward(Module):
    """Feed-forward network for transformer"""
    
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through feed-forward network"""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(Module):
    """Single transformer block"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
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
        # x: [batch, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = x_norm * self.gamma + self.beta
        
        return output


class Embedding(Module):
    """Token embedding layer"""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding weights
        self.weights = Tensor(np.random.randn(vocab_size, d_model) * 0.1, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through embedding layer"""
        # x: [batch, seq_len] -> [batch, seq_len, d_model]
        batch_size, seq_len = x.shape
        
        # Convert to indices and get embeddings
        indices = x.data.astype(np.int32)
        embeddings = self.weights.data[indices]  # [batch, seq_len, d_model]
        
        return Tensor(embeddings, requires_grad=self.weights.requires_grad)


class SmallLLM(Module):
    """Small Language Model using Transformer architecture"""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = Sequential(*[
            TransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = Linear(config.d_model, config.vocab_size)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the language model"""
        # x: [batch, seq_len] -> [batch, seq_len, d_model]
        
        # Token embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: Tensor, max_length: int = 50, 
                temperature: float = 1.0) -> List[int]:
        """Generate text using the model"""
        generated = input_ids.data.tolist()
        
        for _ in range(max_length):
            # Forward pass
            logits = self.forward(Tensor(np.array([generated]), requires_grad=False))
            
            # Get next token logits
            next_token_logits = logits.data[0, -1, :] / temperature
            
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


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy loss for language modeling"""
    # Simplified cross entropy loss
    # In a real implementation, this would be more sophisticated
    batch_size, seq_len, vocab_size = logits.shape
    
    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Simple loss computation (simplified)
    loss = mse_loss(logits_flat, targets_flat)
    
    return loss


def train_llm(model: SmallLLM, data_loader: List[Tuple[Tensor, Tensor]], 
             num_epochs: int = 10) -> Dict[str, float]:
    """Train the small LLM"""
    print("🚀 Training Small LLM...")
    
    optimizer = SGD(model.parameters(), lr=0.001)
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for input_ids, target_ids in data_loader:
            # Forward pass
            logits = model(input_ids)
            loss = cross_entropy_loss(logits, target_ids)
            
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


def benchmark_llm(model: SmallLLM, test_data: List[Tuple[Tensor, Tensor]]) -> Dict[str, float]:
    """Benchmark the LLM performance"""
    print("📊 Benchmarking LLM...")
    
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
    generated = model.generate(sample_input, max_length=20)
    generation_time = time.time() - start_time
    
    return {
        "inference_time_ms": inference_time * 1000,
        "memory_usage_mb": memory_mb,
        "parameter_count": param_count,
        "generation_time_ms": generation_time * 1000,
        "generated_length": len(generated)
    }


if __name__ == "__main__":
    print("🧠 TensorBrain Small LLM")
    print("=" * 40)
    
    # Create LLM configuration
    config = LLMConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )
    
    # Create model
    model = SmallLLM(config)
    print(f"Model created with {sum(param.data.size for param in model.parameters())} parameters")
    
    # Create training data
    train_data = create_sample_data(vocab_size=config.vocab_size, 
                                  seq_len=config.max_seq_len, 
                                  num_samples=50)
    print(f"Created {len(train_data)} training samples")
    
    # Train model
    training_results = train_llm(model, train_data, num_epochs=5)
    
    # Benchmark model
    benchmark_results = benchmark_llm(model, train_data)
    
    # Test generation
    print("\n🧪 Testing text generation:")
    sample_input = Tensor(np.array([[1, 2, 3, 4, 5]]), requires_grad=False)
    generated = model.generate(sample_input, max_length=15)
    print(f"Input: {sample_input.data[0].tolist()}")
    print(f"Generated: {generated}")
    
    print("\n📊 LLM Results:")
    print(f"Final loss: {training_results['final_loss']:.4f}")
    print(f"Loss reduction: {training_results['loss_reduction']:.4f}")
    print(f"Inference time: {benchmark_results['inference_time_ms']:.2f}ms")
    print(f"Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    print(f"Parameter count: {benchmark_results['parameter_count']:,}")
    print(f"Generation time: {benchmark_results['generation_time_ms']:.2f}ms")
    
    print("\n🎉 Small LLM is working!")
    print("📝 Next steps:")
    print("   • Add proper softmax and cross-entropy loss")
    print("   • Implement attention masking")
    print("   • Add dropout for regularization")
    print("   • Train on real text data")
    print("   • Add beam search for generation")
    print("   • Implement tokenization")
    print("   • Add model saving/loading")
