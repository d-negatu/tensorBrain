#!/usr/bin/env python3
"""
Advanced Optimizers for TensorBrain
Adam, RMSprop, and other optimizers
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, mse_loss, SGD


class Adam:
    """Adam optimizer with adaptive learning rates"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize momentum and variance
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
        self.t = 0
        
        print(f"🚀 Initialized Adam optimizer:")
        print(f"   Learning rate: {lr}")
        print(f"   Beta1: {beta1}, Beta2: {beta2}")
        print(f"   Parameters: {len(parameters)}")
    
    def step(self):
        """Perform one optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()


class RMSprop:
    """RMSprop optimizer"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, 
                 alpha: float = 0.99, eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        
        # Initialize running average of squared gradients
        self.v = [np.zeros_like(param.data) for param in parameters]
        
        print(f"🚀 Initialized RMSprop optimizer:")
        print(f"   Learning rate: {lr}")
        print(f"   Alpha: {alpha}")
        print(f"   Parameters: {len(parameters)}")
    
    def step(self):
        """Perform one optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Update running average of squared gradients
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (param.grad ** 2)
                
                # Update parameters
                param.data -= self.lr * param.grad / (np.sqrt(self.v[i]) + self.eps)
    
    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()


class LearningRateScheduler:
    """Learning rate scheduler"""
    
    def __init__(self, optimizer, schedule_type: str = "cosine", 
                 initial_lr: float = 0.001, min_lr: float = 1e-6, 
                 max_epochs: int = 100):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
        print(f"🚀 Initialized {schedule_type} learning rate scheduler:")
        print(f"   Initial LR: {initial_lr}")
        print(f"   Min LR: {min_lr}")
        print(f"   Max epochs: {max_epochs}")
    
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        
        if self.schedule_type == "cosine":
            # Cosine annealing
            lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * self.current_epoch / self.max_epochs)) / 2
        elif self.schedule_type == "linear":
            # Linear decay
            lr = self.initial_lr * (1 - self.current_epoch / self.max_epochs)
        elif self.schedule_type == "exponential":
            # Exponential decay
            lr = self.initial_lr * (0.95 ** self.current_epoch)
        else:
            lr = self.initial_lr
        
        # Update optimizer learning rate
        self.optimizer.lr = max(lr, self.min_lr)
        
        return lr
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.lr


def benchmark_optimizers(model: Module, data_loader: List[Tuple[Tensor, Tensor]], 
                        num_epochs: int = 10) -> Dict[str, Any]:
    """Benchmark different optimizers"""
    print("📊 Benchmarking Optimizers...")
    
    results = {}
    
    # SGD
    print("🔄 Testing SGD...")
    sgd_model = Sequential(*[Linear(2, 4), ReLU(), Linear(4, 2)])
    sgd_optimizer = SGD(sgd_model.parameters(), lr=0.01)
    
    start_time = time.time()
    sgd_losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y in data_loader:
            pred = sgd_model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            sgd_optimizer.step()
            sgd_optimizer.zero_grad()
            epoch_losses.append(loss.data.item())
        sgd_losses.append(np.mean(epoch_losses))
    sgd_time = time.time() - start_time
    
    results["sgd"] = {
        "final_loss": sgd_losses[-1],
        "initial_loss": sgd_losses[0],
        "convergence": sgd_losses[0] - sgd_losses[-1],
        "time": sgd_time
    }
    
    # Adam
    print("🔄 Testing Adam...")
    adam_model = Sequential(*[Linear(2, 4), ReLU(), Linear(4, 2)])
    adam_optimizer = Adam(adam_model.parameters(), lr=0.001)
    
    start_time = time.time()
    adam_losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y in data_loader:
            pred = adam_model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            adam_optimizer.step()
            adam_optimizer.zero_grad()
            epoch_losses.append(loss.data.item())
        adam_losses.append(np.mean(epoch_losses))
    adam_time = time.time() - start_time
    
    results["adam"] = {
        "final_loss": adam_losses[-1],
        "initial_loss": adam_losses[0],
        "convergence": adam_losses[0] - adam_losses[-1],
        "time": adam_time
    }
    
    # RMSprop
    print("🔄 Testing RMSprop...")
    rmsprop_model = Sequential(*[Linear(2, 4), ReLU(), Linear(4, 2)])
    rmsprop_optimizer = RMSprop(rmsprop_model.parameters(), lr=0.001)
    
    start_time = time.time()
    rmsprop_losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y in data_loader:
            pred = rmsprop_model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            rmsprop_optimizer.step()
            rmsprop_optimizer.zero_grad()
            epoch_losses.append(loss.data.item())
        rmsprop_losses.append(np.mean(epoch_losses))
    rmsprop_time = time.time() - start_time
    
    results["rmsprop"] = {
        "final_loss": rmsprop_losses[-1],
        "initial_loss": rmsprop_losses[0],
        "convergence": rmsprop_losses[0] - rmsprop_losses[-1],
        "time": rmsprop_time
    }
    
    return results


if __name__ == "__main__":
    print("🚀 TensorBrain Advanced Optimizers")
    print("=" * 40)
    
    # Create sample data
    from nn import Sequential, Linear, ReLU, mse_loss
    data_loader = []
    for _ in range(20):
        x = Tensor(np.random.randn(10, 2), requires_grad=False)
        y = Tensor(np.random.randn(10, 2), requires_grad=False)
        data_loader.append((x, y))
    
    # Benchmark optimizers
    results = benchmark_optimizers(None, data_loader, num_epochs=5)
    
    print("\n📊 Optimizer Benchmark Results:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name.upper()}:")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Convergence: {result['convergence']:.4f}")
        print(f"  Time: {result['time']:.2f}s")
        print()
    
    # Test learning rate scheduler
    print("🔄 Testing Learning Rate Scheduler...")
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = LearningRateScheduler(optimizer, "cosine", 0.001, 1e-6, 10)
    
    print("Learning rate schedule:")
    for epoch in range(10):
        lr = scheduler.step()
        print(f"Epoch {epoch:2d}: LR = {lr:.6f}")
    
    print("\n🎉 Advanced optimizers are working!")
    print("📝 Next steps:")
    print("   • Add more optimizers (AdaGrad, AdaDelta)")
    print("   • Implement gradient clipping")
    print("   • Add weight decay")
    print("   • Implement warmup scheduling")
    print("   • Add momentum scheduling")
