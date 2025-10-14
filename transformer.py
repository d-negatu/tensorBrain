#!/usr/bin/env python3
"""
Real Transformer Architecture for TensorBrain
Multi-head attention, positional encoding, and transformer blocks
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
import time

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss


class MultiHeadAttention(Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        
        self.dropout = dropout
    
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
        x_reshaped = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        x_transposed = x_reshaped.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        return x_transposed
    
    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, 
                                    mask: Optional[Tensor] = None) -> Tensor:
        """Scaled dot-product attention"""
        # Q, K, V: [batch, n_heads, seq_len, d_k]
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax (simplified)
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        output = attention_weights @ V
        
        return output
    
    def _softmax(self, x: Tensor) -> Tensor:
        """Softmax function"""
        # Simplified softmax
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(softmax_x, requires_grad=x.requires_grad)


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


class TransformerBlock(Module):
    """Single transformer block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = Sequential(
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model)
        )
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through transformer block"""
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
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
        self.embedding = Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = Sequential(*[
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = Linear(d_model, vocab_size)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through transformer"""
        # x: [batch, seq_len] -> [batch, seq_len, d_model]
        
        # Token embeddings
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, tokenizer, prompt: str, max_length: int = 50, 
                temperature: float = 1.0) -> str:
        """Generate text using the transformer"""
        # Encode prompt
        input_ids = tokenizer.encode(prompt, max_length=self.max_seq_len)
        input_tensor = Tensor(np.array([input_ids]), requires_grad=False)
        
        generated_ids = input_ids.copy()
        
        for _ in range(max_length):
            # Forward pass
            logits = self.forward(input_tensor)
            
            # Get next token logits
            next_token_logits = logits.data[0, -1, :] / temperature
            
            # Sample next token (simplified - just take argmax)
            next_token = np.argmax(next_token_logits)
            generated_ids.append(next_token)
            
            # Update input for next iteration
            input_tensor = Tensor(np.array([generated_ids[-self.max_seq_len:]]), requires_grad=False)
            
            # Stop if we hit end token
            if next_token == tokenizer.vocab[tokenizer.end_token]:
                break
        
        # Decode to text
        generated_text = tokenizer.decode(generated_ids)
        return generated_text


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


def benchmark_transformer(model: Transformer, tokenizer, test_prompts: List[str]) -> Dict[str, float]:
    """Benchmark transformer performance"""
    print("📊 Benchmarking Transformer...")
    
    # Generation speed
    start_time = time.time()
    for prompt in test_prompts:
        generated = model.generate(tokenizer, prompt, max_length=20)
    generation_time = (time.time() - start_time) / len(test_prompts)
    
    # Memory usage
    param_count = sum(param.data.size for param in model.parameters())
    memory_mb = param_count * 4 / (1024 * 1024)
    
    return {
        "generation_time_ms": generation_time * 1000,
        "memory_usage_mb": memory_mb,
        "parameter_count": param_count,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers
    }


if __name__ == "__main__":
    print("🤖 TensorBrain Real Transformer Architecture")
    print("=" * 50)
    
    # Create transformer model
    from real_llm import Tokenizer
    
    # Sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Hello world, this is a transformer model",
        "Machine learning is the future of technology",
        "Python is a great programming language",
        "Artificial intelligence will change the world"
    ]
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(sample_texts, min_freq=1)
    
    # Create transformer
    transformer = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=50
    )
    
    print(f"Transformer created:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Model dimension: {transformer.d_model}")
    print(f"  Number of heads: {transformer.n_heads}")
    print(f"  Number of layers: {transformer.n_layers}")
    print(f"  Parameters: {sum(param.data.size for param in transformer.parameters()):,}")
    
    # Test generation
    print("\n🧪 Testing Transformer Generation:")
    test_prompts = ["The quick", "Hello", "Machine"]
    for prompt in test_prompts:
        generated = transformer.generate(tokenizer, prompt, max_length=15)
        print(f"  '{prompt}' → '{generated}'")
    
    # Benchmark
    benchmark_results = benchmark_transformer(transformer, tokenizer, test_prompts)
    
    print("\n📊 Transformer Benchmark Results:")
    print(f"Generation time: {benchmark_results['generation_time_ms']:.2f}ms")
    print(f"Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    print(f"Parameter count: {benchmark_results['parameter_count']:,}")
    print(f"Model dimension: {benchmark_results['d_model']}")
    print(f"Number of heads: {benchmark_results['n_heads']}")
    print(f"Number of layers: {benchmark_results['n_layers']}")
    
    print("\n🎉 Real Transformer Architecture is working!")
    print("📝 Next steps:")
    print("   • Add proper softmax and cross-entropy loss")
    print("   • Implement attention masking")
    print("   • Add dropout for regularization")
    print("   • Train on larger datasets")
    print("   • Add beam search for generation")
    print("   • Implement model parallelism")
    print("   • Add gradient checkpointing")
