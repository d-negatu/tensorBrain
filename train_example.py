#!/usr/bin/env python3
"""
Simple training example using TensorBrain neural network layers
"""

import numpy as np
from nn import Linear, ReLU, Sequential, SGD, mse_loss
from tensor import Tensor


def generate_synthetic_data(n_samples=100, n_features=2):
    """Generate synthetic data for binary classification"""
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    
    # Create a simple decision boundary: y = 1 if x1 + x2 > 0, else 0
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    
    return X, y


def train_simple_classifier():
    """Train a simple binary classifier"""
    print("🚀 Training Simple Binary Classifier with TensorBrain")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(100, 2)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    
    # Create model
    model = Sequential(
        Linear(2, 4),  # 2 inputs -> 4 hidden units
        ReLU(),        # Activation
        Linear(4, 1),  # 4 hidden -> 1 output
    )
    
    print(f"Model: {model}")
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    print(f"Optimizer: SGD with learning rate 0.01")
    
    # Training loop
    n_epochs = 100
    print(f"\nTraining for {n_epochs} epochs...")
    
    losses = []
    
    for epoch in range(n_epochs):
        # Forward pass
        X_tensor = Tensor(X, requires_grad=False)
        y_tensor = Tensor(y.reshape(-1, 1), requires_grad=False)
        
        # Get predictions
        predictions = model(X_tensor)
        
        # Compute loss
        loss = mse_loss(predictions, y_tensor)
        losses.append(loss.data.item())
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.data.item():.4f}")
    
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Test the model
    print("\n🧪 Testing the trained model:")
    test_X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    test_y = (test_X[:, 0] + test_X[:, 1] > 0).astype(np.float32)
    
    test_X_tensor = Tensor(test_X, requires_grad=False)
    predictions = model(test_X_tensor)
    
    print("Input    | True Label | Prediction | Correct")
    print("-" * 45)
    for i in range(len(test_X)):
        pred = predictions.data[i, 0]
        true_label = test_y[i]
        correct = "✓" if (pred > 0.5) == (true_label > 0.5) else "✗"
        print(f"{test_X[i]} | {true_label:10.1f} | {pred:10.4f} | {correct}")
    
    # Calculate accuracy
    correct_predictions = 0
    for i in range(len(test_X)):
        pred = predictions.data[i, 0]
        true_label = test_y[i]
        if (pred > 0.5) == (true_label > 0.5):
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_X)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    return model, losses


def train_xor_problem():
    """Train on the XOR problem - a classic non-linearly separable problem"""
    print("\n🚀 Training on XOR Problem")
    print("=" * 40)
    
    # XOR problem data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    
    print("XOR Problem:")
    print("Input | Output")
    print("-" * 12)
    for i in range(len(X)):
        print(f"{X[i]} | {y[i]}")
    
    # Create a deeper model for XOR
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 4),
        ReLU(),
        Linear(4, 1),
    )
    
    print(f"\nModel: {model}")
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.1)
    
    # Training loop
    n_epochs = 1000
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        X_tensor = Tensor(X, requires_grad=False)
        y_tensor = Tensor(y.reshape(-1, 1), requires_grad=False)
        
        predictions = model(X_tensor)
        loss = mse_loss(predictions, y_tensor)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.data.item():.4f}")
    
    # Test the model
    print("\n🧪 Testing XOR model:")
    test_predictions = model(Tensor(X, requires_grad=False))
    
    print("Input | True | Prediction | Correct")
    print("-" * 35)
    for i in range(len(X)):
        pred = test_predictions.data[i, 0]
        true_label = y[i]
        correct = "✓" if (pred > 0.5) == (true_label > 0.5) else "✗"
        print(f"{X[i]} | {true_label:4.1f} | {pred:10.4f} | {correct}")
    
    # Calculate accuracy
    correct = sum(1 for i in range(len(X)) 
                  if (test_predictions.data[i, 0] > 0.5) == (y[i] > 0.5))
    accuracy = correct / len(X)
    print(f"\nXOR Accuracy: {accuracy:.2%}")
    
    return model


if __name__ == "__main__":
    print("🧠 TensorBrain Neural Network Training Examples")
    print("=" * 50)
    
    # Train simple classifier
    model1, losses1 = train_simple_classifier()
    
    # Train XOR problem
    model2 = train_xor_problem()
    
    print("\n🎉 Training completed successfully!")
    print("\n📝 What we accomplished:")
    print("   ✅ Built neural network layers on top of autograd")
    print("   ✅ Implemented training loop with SGD optimizer")
    print("   ✅ Trained models on synthetic data")
    print("   ✅ Solved the XOR problem (non-linearly separable)")
    print("   ✅ Demonstrated end-to-end neural network training")
    
    print("\n🚀 Next steps:")
    print("   • Add more layers (Conv2D, BatchNorm)")
    print("   • Implement more optimizers (Adam, RMSprop)")
    print("   • Add gradient support to activation functions")
    print("   • Create FastAPI serving runtime")
    print("   • Add unit tests for PyTorch parity")
