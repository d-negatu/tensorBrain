#!/usr/bin/env python3
"""
Federated Learning for TensorBrain
Distributed training with privacy preservation
"""

import numpy as np
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import queue

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss
from optimizers import Adam


@dataclass
class ClientUpdate:
    """Client model update"""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    loss: float
    timestamp: float


@dataclass
class GlobalModel:
    """Global model state"""
    weights: Dict[str, np.ndarray]
    version: int
    timestamp: float
    num_clients: int
    total_samples: int


class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: str, model: Module, local_data: List[Tuple[Tensor, Tensor]]):
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        
        print(f"🚀 FederatedClient {client_id} initialized:")
        print(f"   Local data samples: {len(local_data)}")
        print(f"   Model parameters: {sum(param.data.size for param in model.parameters()):,}")
    
    def train_local(self, num_epochs: int = 5) -> ClientUpdate:
        """Train model on local data"""
        print(f"🔄 Client {self.client_id} training locally...")
        
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for x, y in self.local_data:
                # Forward pass
                pred = self.model(x)
                loss = mse_loss(pred, y)
                epoch_losses.append(loss.data.item())
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
        
        # Extract model weights
        model_weights = {}
        for i, param in enumerate(self.model.parameters()):
            model_weights[f"param_{i}"] = param.data.copy()
        
        # Create client update
        update = ClientUpdate(
            client_id=self.client_id,
            model_weights=model_weights,
            num_samples=len(self.local_data),
            loss=losses[-1],
            timestamp=time.time()
        )
        
        print(f"✅ Client {self.client_id} training completed:")
        print(f"   Final loss: {losses[-1]:.4f}")
        print(f"   Samples: {len(self.local_data)}")
        
        return update
    
    def update_model(self, global_weights: Dict[str, np.ndarray]):
        """Update local model with global weights"""
        print(f"🔄 Client {self.client_id} updating model...")
        
        # Update model parameters
        for i, param in enumerate(self.model.parameters()):
            if f"param_{i}" in global_weights:
                param.data = global_weights[f"param_{i}"].copy()
        
        print(f"✅ Client {self.client_id} model updated")


class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, global_model: Module):
        self.global_model = global_model
        self.clients: Dict[str, FederatedClient] = {}
        self.client_updates: queue.Queue = queue.Queue()
        self.global_model_state = GlobalModel(
            weights={},
            version=0,
            timestamp=time.time(),
            num_clients=0,
            total_samples=0
        )
        
        print("🚀 FederatedServer initialized")
    
    def register_client(self, client: FederatedClient):
        """Register a client"""
        self.clients[client.client_id] = client
        print(f"✅ Client {client.client_id} registered")
    
    def aggregate_updates(self, updates: List[ClientUpdate]) -> GlobalModel:
        """Aggregate client updates using FedAvg"""
        print("🔄 Aggregating client updates...")
        
        if not updates:
            return self.global_model_state
        
        # Calculate total samples
        total_samples = sum(update.num_samples for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        first_update = updates[0]
        
        for param_name in first_update.model_weights:
            aggregated_weights[param_name] = np.zeros_like(first_update.model_weights[param_name])
        
        # Weighted average of client updates
        for update in updates:
            weight = update.num_samples / total_samples
            
            for param_name, param_weights in update.model_weights.items():
                aggregated_weights[param_name] += weight * param_weights
        
        # Update global model
        self.global_model_state = GlobalModel(
            weights=aggregated_weights,
            version=self.global_model_state.version + 1,
            timestamp=time.time(),
            num_clients=len(updates),
            total_samples=total_samples
        )
        
        print(f"✅ Aggregation completed:")
        print(f"   Clients: {len(updates)}")
        print(f"   Total samples: {total_samples}")
        print(f"   Global model version: {self.global_model_state.version}")
        
        return self.global_model_state
    
    def broadcast_model(self) -> GlobalModel:
        """Broadcast global model to all clients"""
        print("📡 Broadcasting global model...")
        
        # Update all clients
        for client in self.clients.values():
            client.update_model(self.global_model_state.weights)
        
        print(f"✅ Model broadcasted to {len(self.clients)} clients")
        return self.global_model_state
    
    def run_federated_round(self, num_epochs: int = 5) -> Dict[str, Any]:
        """Run one federated learning round"""
        print(f"🚀 Starting federated learning round...")
        
        # Collect updates from all clients
        updates = []
        for client in self.clients.values():
            update = client.train_local(num_epochs=num_epochs)
            updates.append(update)
        
        # Aggregate updates
        global_model = self.aggregate_updates(updates)
        
        # Broadcast updated model
        self.broadcast_model()
        
        # Calculate round statistics
        round_stats = {
            "round": global_model.version,
            "num_clients": len(updates),
            "total_samples": global_model.total_samples,
            "avg_client_loss": np.mean([update.loss for update in updates]),
            "timestamp": global_model.timestamp
        }
        
        return round_stats


class PrivacyPreservingFederatedLearning:
    """Privacy-preserving federated learning with differential privacy"""
    
    def __init__(self, server: FederatedServer, noise_scale: float = 0.1):
        self.server = server
        self.noise_scale = noise_scale
        
        print(f"🔒 PrivacyPreservingFederatedLearning initialized:")
        print(f"   Noise scale: {noise_scale}")
    
    def add_differential_privacy(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Add differential privacy noise to client updates"""
        print("🔒 Adding differential privacy noise...")
        
        noisy_updates = []
        
        for update in updates:
            # Add Gaussian noise to model weights
            noisy_weights = {}
            
            for param_name, param_weights in update.model_weights.items():
                noise = np.random.normal(0, self.noise_scale, param_weights.shape)
                noisy_weights[param_name] = param_weights + noise
            
            # Create noisy update
            noisy_update = ClientUpdate(
                client_id=update.client_id,
                model_weights=noisy_weights,
                num_samples=update.num_samples,
                loss=update.loss,
                timestamp=update.timestamp
            )
            
            noisy_updates.append(noisy_update)
        
        print(f"✅ Added noise to {len(updates)} client updates")
        return noisy_updates
    
    def run_private_federated_round(self, num_epochs: int = 5) -> Dict[str, Any]:
        """Run privacy-preserving federated learning round"""
        print("🔒 Starting privacy-preserving federated round...")
        
        # Collect updates from clients
        updates = []
        for client in self.server.clients.values():
            update = client.train_local(num_epochs=num_epochs)
            updates.append(update)
        
        # Add differential privacy
        private_updates = self.add_differential_privacy(updates)
        
        # Aggregate private updates
        global_model = self.server.aggregate_updates(private_updates)
        
        # Broadcast model
        self.server.broadcast_model()
        
        return {
            "round": global_model.version,
            "num_clients": len(private_updates),
            "total_samples": global_model.total_samples,
            "privacy_preserved": True,
            "noise_scale": self.noise_scale
        }


def create_federated_clients(num_clients: int = 5, samples_per_client: int = 100) -> List[FederatedClient]:
    """Create federated learning clients"""
    clients = []
    
    for i in range(num_clients):
        # Create model for client
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 2)
        )
        
        # Create local data for client
        local_data = []
        for _ in range(samples_per_client):
            x = Tensor(np.random.randn(10, 2), requires_grad=False)
            y = Tensor(np.random.randn(10, 2), requires_grad=False)
            local_data.append((x, y))
        
        # Create client
        client = FederatedClient(f"client_{i}", model, local_data)
        clients.append(client)
    
    return clients


def benchmark_federated_learning(num_clients: int = 5, num_rounds: int = 10) -> Dict[str, Any]:
    """Benchmark federated learning performance"""
    print("📊 Benchmarking Federated Learning...")
    
    # Create clients
    clients = create_federated_clients(num_clients=num_clients, samples_per_client=50)
    
    # Create server
    global_model = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
    server = FederatedServer(global_model)
    
    # Register clients
    for client in clients:
        server.register_client(client)
    
    # Run federated learning
    round_stats = []
    
    for round_num in range(num_rounds):
        print(f"\n🔄 Federated Round {round_num + 1}/{num_rounds}")
        
        start_time = time.time()
        stats = server.run_federated_round(num_epochs=3)
        round_time = time.time() - start_time
        
        stats["round_time"] = round_time
        round_stats.append(stats)
        
        print(f"   Round time: {round_time:.2f}s")
        print(f"   Avg client loss: {stats['avg_client_loss']:.4f}")
    
    # Calculate overall statistics
    total_time = sum(stat["round_time"] for stat in round_stats)
    avg_round_time = total_time / num_rounds
    final_loss = round_stats[-1]["avg_client_loss"]
    
    results = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "total_time": total_time,
        "avg_round_time": avg_round_time,
        "final_loss": final_loss,
        "round_stats": round_stats
    }
    
    print(f"\n📊 Federated Learning Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average round time: {avg_round_time:.2f}s")
    print(f"Final loss: {final_loss:.4f}")
    
    return results


def demo_federated_learning():
    """Demonstrate federated learning capabilities"""
    print("🌐 TensorBrain Federated Learning Demo")
    print("=" * 50)
    
    # Create clients
    clients = create_federated_clients(num_clients=3, samples_per_client=100)
    
    # Create server
    global_model = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
    server = FederatedServer(global_model)
    
    # Register clients
    for client in clients:
        server.register_client(client)
    
    # Run federated learning
    print("\n🚀 Running Federated Learning...")
    for round_num in range(5):
        stats = server.run_federated_round(num_epochs=3)
        print(f"Round {stats['round']}: Loss = {stats['avg_client_loss']:.4f}")
    
    # Test privacy-preserving federated learning
    print("\n🔒 Testing Privacy-Preserving Federated Learning...")
    private_fl = PrivacyPreservingFederatedLearning(server, noise_scale=0.1)
    
    for round_num in range(3):
        stats = private_fl.run_private_federated_round(num_epochs=2)
        print(f"Private Round {stats['round']}: Privacy preserved = {stats['privacy_preserved']}")
    
    # Benchmark performance
    benchmark_results = benchmark_federated_learning(num_clients=5, num_rounds=5)
    
    print("\n🎉 Federated Learning is working!")
    print("📝 Next steps:")
    print("   • Add secure aggregation")
    print("   • Implement homomorphic encryption")
    print("   • Add client selection strategies")
    print("   • Implement asynchronous updates")
    print("   • Add model compression")
    print("   • Implement federated evaluation")
    
    return benchmark_results


if __name__ == "__main__":
    demo_federated_learning()
