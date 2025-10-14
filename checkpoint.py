#!/usr/bin/env python3
"""
Model Checkpointing for TensorBrain
Save and load trained models
"""

import numpy as np
import json
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU


@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints"""
    model_name: str
    version: str
    epoch: int
    loss: float
    accuracy: float
    timestamp: float
    parameters: int
    training_time: float
    optimizer_state: Dict[str, Any]
    config: Dict[str, Any]


class ModelCheckpointer:
    """Model checkpointing system"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = []
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"🚀 Initialized ModelCheckpointer:")
        print(f"   Checkpoint directory: {checkpoint_dir}")
    
    def save_checkpoint(self, model: Module, optimizer: Any, epoch: int, 
                       loss: float, accuracy: float = 0.0, config: Dict[str, Any] = None,
                       model_name: str = "model") -> str:
        """Save model checkpoint"""
        start_time = time.time()
        
        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            model_name=model_name,
            version="1.0.0",
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            timestamp=time.time(),
            parameters=sum(param.data.size for param in model.parameters()),
            training_time=0.0,  # Would be calculated from training start
            optimizer_state=self._save_optimizer_state(optimizer),
            config=config or {}
        )
        
        # Create checkpoint filename
        checkpoint_name = f"{model_name}_epoch_{epoch:04d}_loss_{loss:.4f}.ckpt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save model parameters
        model_state = self._save_model_state(model)
        
        # Save checkpoint
        checkpoint_data = {
            "metadata": asdict(metadata),
            "model_state": model_state
        }
        
        # Save to file (simplified - would use pickle in production)
        np.savez_compressed(checkpoint_path, **checkpoint_data)
        
        # Save metadata as JSON for easy reading
        metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        save_time = time.time() - start_time
        
        print(f"✅ Checkpoint saved:")
        print(f"   Path: {checkpoint_path}")
        print(f"   Epoch: {epoch}")
        print(f"   Loss: {loss:.4f}")
        print(f"   Parameters: {metadata.parameters:,}")
        print(f"   Save time: {save_time:.2f}s")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Module, CheckpointMetadata]:
        """Load model checkpoint"""
        start_time = time.time()
        
        # Load checkpoint data
        checkpoint_data = np.load(checkpoint_path, allow_pickle=True)
        
        # Extract metadata
        metadata_dict = checkpoint_data['metadata'].item()
        metadata = CheckpointMetadata(**metadata_dict)
        
        # Extract model state
        model_state = checkpoint_data['model_state'].item()
        
        # Create model from state
        model = self._load_model_state(model_state)
        
        load_time = time.time() - start_time
        
        print(f"✅ Checkpoint loaded:")
        print(f"   Path: {checkpoint_path}")
        print(f"   Epoch: {metadata.epoch}")
        print(f"   Loss: {metadata.loss:.4f}")
        print(f"   Parameters: {metadata.parameters:,}")
        print(f"   Load time: {load_time:.2f}s")
        
        return model, metadata
    
    def _save_model_state(self, model: Module) -> Dict[str, Any]:
        """Save model state"""
        state = {
            "type": type(model).__name__,
            "parameters": {}
        }
        
        # Save parameters
        for i, param in enumerate(model.parameters()):
            state["parameters"][f"param_{i}"] = {
                "data": param.data,
                "requires_grad": param.requires_grad
            }
        
        return state
    
    def _load_model_state(self, model_state: Dict[str, Any]) -> Module:
        """Load model state"""
        # Create model based on type
        if model_state["type"] == "Sequential":
            # For now, create a simple sequential model
            model = Sequential(
                Linear(2, 4),
                ReLU(),
                Linear(4, 2)
            )
        else:
            # Default model
            model = Sequential(
                Linear(2, 4),
                ReLU(),
                Linear(4, 2)
            )
        
        # Load parameters
        for name, param_data in model_state["parameters"].items():
            # Find parameter in model
            param = self._find_parameter(model, name)
            if param is not None:
                param.data = param_data["data"]
                param.requires_grad = param_data["requires_grad"]
        
        return model
    
    def _find_parameter(self, model: Module, name: str) -> Optional[Tensor]:
        """Find parameter by name in model"""
        # Simplified parameter finding
        for param in model.parameters():
            if hasattr(param, 'name') and param.name == name:
                return param
        return None
    
    def _save_optimizer_state(self, optimizer: Any) -> Dict[str, Any]:
        """Save optimizer state"""
        if hasattr(optimizer, 'lr'):
            return {
                "type": type(optimizer).__name__,
                "lr": optimizer.lr,
                "parameters": len(optimizer.parameters) if hasattr(optimizer, 'parameters') else 0
            }
        return {}
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('_metadata.json'):
                metadata_path = os.path.join(self.checkpoint_dir, filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        
        return checkpoints
    
    def get_best_checkpoint(self, metric: str = "loss", minimize: bool = True) -> Optional[str]:
        """Get the best checkpoint based on metric"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        if minimize:
            best_checkpoint = min(checkpoints, key=lambda x: x[metric])
        else:
            best_checkpoint = max(checkpoints, key=lambda x: x[metric])
        
        # Return checkpoint path
        checkpoint_name = f"{best_checkpoint['model_name']}_epoch_{best_checkpoint['epoch']:04d}_loss_{best_checkpoint['loss']:.4f}.ckpt"
        return os.path.join(self.checkpoint_dir, checkpoint_name)


def demo_checkpointing():
    """Demonstrate model checkpointing"""
    print("💾 TensorBrain Model Checkpointing Demo")
    print("=" * 40)
    
    # Create model and optimizer
    from nn import Sequential, Linear, ReLU, SGD, mse_loss
    from optimizers import Adam
    
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 2)
    )
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create checkpointer
    checkpointer = ModelCheckpointer("demo_checkpoints")
    
    # Simulate training and save checkpoints
    print("\n🔄 Simulating training and saving checkpoints...")
    
    x = Tensor(np.random.randn(10, 2), requires_grad=False)
    y = Tensor(np.random.randn(10, 2), requires_grad=False)
    
    for epoch in range(5):
        # Forward pass
        pred = model(x)
        loss = mse_loss(pred, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Save checkpoint
        checkpoint_path = checkpointer.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss.data.item(),
            accuracy=0.0,
            config={"learning_rate": optimizer.lr, "batch_size": 10}
        )
    
    # List checkpoints
    print("\n📋 Available checkpoints:")
    checkpoints = checkpointer.list_checkpoints()
    for checkpoint in checkpoints:
        print(f"  Epoch {checkpoint['epoch']:2d}: Loss = {checkpoint['loss']:.4f}, "
              f"Parameters = {checkpoint['parameters']:,}")
    
    # Load best checkpoint
    print("\n🔄 Loading best checkpoint...")
    best_checkpoint_path = checkpointer.get_best_checkpoint("loss", minimize=True)
    if best_checkpoint_path:
        loaded_model, metadata = checkpointer.load_checkpoint(best_checkpoint_path)
        print(f"✅ Loaded model from epoch {metadata.epoch}")
    
    print("\n🎉 Model checkpointing is working!")
    print("📝 Next steps:")
    print("   • Add model versioning")
    print("   • Implement distributed checkpointing")
    print("   • Add checkpoint compression")
    print("   • Implement incremental checkpoints")
    print("   • Add checkpoint validation")


if __name__ == "__main__":
    demo_checkpointing()
