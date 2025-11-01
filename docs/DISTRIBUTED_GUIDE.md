# Distributed Training Guide

## Overview

This guide covers distributed training with TensorBrain across multiple GPUs and machines.

## Prerequisites

- Multiple GPUs or machines
- NCCL (NVIDIA Collective Communications Library) for GPU communication
- Gloo backend for CPU communication

## Data Parallel Training

### Single Machine, Multiple GPUs

```python
import tensorbrain as tb
from tb.nn.parallel import DataParallel

model = tb.nn.ResNet50()
device = 'cuda' if tb.cuda.is_available() else 'cpu'
model = model.to(device)

# Wrap model for data parallelism
if tb.cuda.device_count() > 1:
    model = DataParallel(model)
    print(f"Using {tb.cuda.device_count()} GPUs")

# Training loop remains the same
criterion = tb.nn.CrossEntropyLoss()
optimizer = tb.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Distributed Data Parallel

### Multi-GPU, Single Machine

```python
import tb
import tb.distributed as dist
from tb.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(
    backend='nccl',  # or 'gloo'
    world_size=4,
    rank=rank
)

# Set device based on rank
torch.cuda.set_device(rank)

# Create model
model = tb.nn.ResNet50()
model = model.to(rank)
model = DDP(model, device_ids=[rank])

# Create sampler for distributed data loading
train_sampler = tb.utils.data.DistributedSampler(
    train_dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True
)

train_loader = tb.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler
)

# Training loop
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
    
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(rank)
        batch_labels = batch_labels.to(rank)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Launching Distributed Training

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py --args

# Using Python with environment variables
PYTHON_MULTIPROCESSING_METHOD=spawn python -m torch.distributed.launch \
    --nproc_per_node=4 train.py --args
```

## Multi-Machine Training

```python
# On machine 0 (master)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=8
torchrun --nproc_per_node=4 train.py

# On machine 1 (worker)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export RANK=4
export WORLD_SIZE=8
torchrun --nproc_per_node=4 train.py
```

## Gradient Accumulation

```python
accumulation_steps = 4

for epoch in range(num_epochs):
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels) / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

## Synchronization and Barriers

```python
# Synchronize all processes
dist.barrier()

# All-reduce operations
if dist.get_rank() == 0:
    tensor = tb.tensor([1.0, 2.0, 3.0])
else:
    tensor = tb.tensor([0.0, 0.0, 0.0])

# Sum across all ranks
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {dist.get_rank()}: {tensor}")  # All ranks see [1.0, 2.0, 3.0]
```

## Communication Collectives

```python
# All-gather
tensor_list = [tb.empty_like(tensor) for _ in range(dist.get_world_size())]
dist.all_gather(tensor_list, tensor)

# Broadcast
if dist.get_rank() == 0:
    broadcast_tensor = tb.tensor([1.0, 2.0, 3.0])
else:
    broadcast_tensor = tb.zeros(3)

dist.broadcast(broadcast_tensor, src=0)

# Scatter
if dist.get_rank() == 0:
    scatter_list = [tb.tensor([1.0]), tb.tensor([2.0]), tb.tensor([3.0])]
else:
    scatter_list = [None]

tensor = tb.empty(1)
dist.scatter(tensor, scatter_list if dist.get_rank() == 0 else None, src=0)
```

## Best Practices

1. **Batch Size**: Adjust batch size per GPU, not total batch size
2. **Learning Rate**: Scale learning rate by number of GPUs
3. **Gradient Sync**: Use `no_sync()` to reduce communication overhead
4. **Checkpointing**: Only save from rank 0 to avoid conflicts

## Troubleshooting

- **Timeout Issues**: Increase timeout value in `init_process_group()`
- **Hanging Processes**: Ensure all ranks reach synchronization points
- **Communication Errors**: Check network connectivity and NCCL settings

## See Also

- [API Reference](./API_REFERENCE.md)
- [Quantization Guide](./QUANTIZATION_GUIDE.md)
- [Contributing Guide](../CONTRIBUTING.md)
